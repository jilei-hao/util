import vtk
import numpy as np
from sklearn.decomposition import PCA
from vtk.util import numpy_support
import SimpleITK as sitk

def add_label_as_point_data(model, label):
  vtk_array = vtk.vtkIntArray()
  vtk_array.SetName("Label")
  vtk_array.SetNumberOfComponents(1)
  vtk_array.SetNumberOfTuples(model.GetNumberOfPoints())
  for i in range(model.GetNumberOfPoints()):
    vtk_array.SetValue(i, label)
  model.GetPointData().AddArray(vtk_array)


def append_unstructured_grid(model_list):
  filter_append = vtk.vtkAppendFilter()
  for model in model_list:
    filter_append.AddInputData(model)
  filter_append.Update()
  return filter_append.GetOutput()


def convert_unstructured_grid_to_polydata(unstructured_grid):
  """
  Convert an unstructured grid to a polydata object.
  """
  geometry_filter = vtk.vtkGeometryFilter()
  geometry_filter.SetInputData(unstructured_grid)
  geometry_filter.Update()
  polydata = geometry_filter.GetOutput()
  return polydata


def readPolyData(filename):
  """Read polydata from file.
  Args:
    filename: path to file
  Returns:
    polydata: vtkPolyData
  """

  if filename.endswith('.vtp'):
    reader = vtk.vtkXMLPolyDataReader()
  elif filename.endswith('.vtk'):
    reader = vtk.vtkPolyDataReader()
  elif filename.endswith('.stl'):
    reader = vtk.vtkSTLReader()
  else:
    raise ValueError('Unsupported file format')
  
  reader.SetFileName(filename)
  reader.Update()
  polydata = reader.GetOutput()
  
  return polydata


def writePolyData(polydata, filename):
  """Write polydata to file.
  Args:
    polydata: vtkPolyData
    filename: path to file
  """
  if filename.endswith('.vtp'):
    writer = vtk.vtkXMLPolyDataWriter()
  elif filename.endswith('.vtk'):
    writer = vtk.vtkPolyDataWriter()
  elif filename.endswith('.stl'):
    writer = vtk.vtkSTLWriter()
  else:
    raise ValueError('Unsupported file format')
  
  writer.SetFileName(filename)
  writer.SetInputData(polydata)
  writer.Write()


def read_unstructured_grid(filename):
  """Read unstructured grid from file.
  Args:
    filename: path to file
  Returns:
    unstructured_grid: vtkUnstructuredGrid
  """
  if filename.endswith('.vtu'):
    reader = vtk.vtkXMLUnstructuredGridReader()
  elif filename.endswith('.vtk'):
    reader = vtk.vtkUnstructuredGridReader()
  else:
    raise ValueError('Unsupported file format')
  
  reader.SetFileName(filename)
  reader.Update()
  unstructured_grid = reader.GetOutput()
  
  return unstructured_grid


def write_unstructured_grid(unstructured_grid, filename):
  """Write unstructured grid to file.
  Args:
    unstructured_grid: vtkUnstructuredGrid
    filename: path to file
  """
  if filename.endswith('.vtu'):
    writer = vtk.vtkXMLUnstructuredGridWriter()
  elif filename.endswith('.vtk'):
    writer = vtk.vtkUnstructuredGridWriter()
  else:
    raise ValueError('Unsupported file format')
  
  writer.SetFileName(filename)
  writer.SetInputData(unstructured_grid)
  writer.Write()

def applyTransform(polydata, matrix):
  """Apply transformation matrix to polydata.
  Args:
    polydata: vtkPolyData
    matrix: vtkMatrix4x4
  Returns:
    polydata: vtkPolyData
  """
  transform = vtk.vtkTransform()
  transform.SetMatrix(matrix)
  
  transform_filter = vtk.vtkTransformPolyDataFilter()
  transform_filter.SetInputData(polydata)
  transform_filter.SetTransform(transform)
  transform_filter.Update()
  
  return transform_filter.GetOutput()


def generateNormals(polydata):
  """Generate normals for polydata.
  Args:
    polydata: vtkPolyData
  Returns:
    polydata: vtkPolyData
  """
  normals = vtk.vtkPolyDataNormals()
  normals.SetInputData(polydata)
  normals.ComputePointNormalsOn()
  normals.ComputeCellNormalsOn()
  normals.AutoOrientNormalsOn()
  normals.ConsistencyOn()
  normals.Update()
  
  return normals.GetOutput()



def smoothSurfaceBoundary(polydata, num_iterations=20, passband=0.001, feature_angle=120.0):
  """Smooth boundary of polydata
  Args:
    polydata: vtkPolyData
    num_iterations: int
    passband: float
    feature_angle: float
  Returns:
    polydata: vtkPolyData
  """

  smooth = vtk.vtkWindowedSincPolyDataFilter()
  smooth.SetInputData(polydata)
  smooth.SetNumberOfIterations(num_iterations)
  smooth.BoundarySmoothingOn()
  smooth.FeatureEdgeSmoothingOff()
  smooth.SetFeatureAngle(feature_angle)
  smooth.SetPassBand(passband)
  smooth.NonManifoldSmoothingOn()
  smooth.NormalizeCoordinatesOn()
  smooth.Update()
  
  return smooth.GetOutput()

def filterCellsByNormal(polydata, fn):
  """Filter cells from polydata based on cell data.
  Args:
    polydata: vtkPolyData
    fn: function(x, y, z) => boolean
  Returns:
    polydata: vtkPolyData
  """

  polydata = generateNormals(polydata)

  normals = polydata.GetCellData().GetNormals()
  
  filtered = vtk.vtkCellArray()

  for i in range(polydata.GetNumberOfCells()):
    normal = normals.GetTuple(i)
    if fn(normal[0], normal[1], normal[2]):
      filtered.InsertNextCell(polydata.GetCell(i))
  
  filtered_polydata = vtk.vtkPolyData()
  filtered_polydata.SetPoints(polydata.GetPoints())
  filtered_polydata.SetPolys(filtered)
  
  return filtered_polydata


def getCentroid(polydata):
  """Get centroid of polydata.
  Args:
    polydata: vtkPolyData
  Returns:
    centroid: tuple
  """
  points = polydata.GetPoints()
  num_points = points.GetNumberOfPoints()
  
  centroid = [0, 0, 0]
  for i in range(num_points):
    point = points.GetPoint(i)
    centroid[0] += point[0]
    centroid[1] += point[1]
    centroid[2] += point[2]
  
  centroid[0] /= num_points
  centroid[1] /= num_points
  centroid[2] /= num_points
  
  return tuple(centroid)

def getAverageNormalVector(polydata):
  """Get average normal vector from polydata.
  Args:
    polydata: vtkPolyData
  Returns:
    normal: tuple
  """
  polydata = generateNormals(polydata)
  normals = polydata.GetCellData().GetNormals()
  
  avg_normal = [0, 0, 0]
  for i in range(polydata.GetNumberOfCells()):
    normal = normals.GetTuple(i)
    avg_normal[0] += normal[0]
    avg_normal[1] += normal[1]
    avg_normal[2] += normal[2]
  
  avg_normal[0] /= polydata.GetNumberOfCells()
  avg_normal[1] /= polydata.GetNumberOfCells()
  avg_normal[2] /= polydata.GetNumberOfCells()
  
  return avg_normal


def remesh(polydata):
  """Remesh polydata by recreating triangles from the points.
  Args:
    polydata: vtkPolyData
  Returns:
    polydata: vtkPolyData
  """
  # Extract points from the input polydata
  points = polydata.GetPoints()

  # Create a new polydata to hold the points
  point_polydata = vtk.vtkPolyData()
  point_polydata.SetPoints(points)

  # Perform Delaunay triangulation
  delaunay = vtk.vtkDelaunay2D()
  delaunay.SetInputData(point_polydata)
  delaunay.Update()

  return delaunay.GetOutput()


def linearSubdivision(polydata, num_subdivisions=3):
  """Subdivide polydata using linear subdivision.
  Args:
    polydata: vtkPolyData
    num_subdivisions: int
  Returns:
    polydata: vtkPolyData
  """
  fltLinarSub = vtk.vtkLinearSubdivisionFilter()
  fltLinarSub.SetInputData(polydata)
  fltLinarSub.SetNumberOfSubdivisions(num_subdivisions)
  fltLinarSub.Update()
  return fltLinarSub.GetOutput()




def extractLargestRegion(polydata):
  connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
  connectivity_filter.SetInputData(polydata)
  connectivity_filter.SetExtractionModeToLargestRegion()
  connectivity_filter.Update()
  return connectivity_filter.GetOutput()


def cleanPolyData(polydata):
  cleanFilter = vtk.vtkCleanPolyData()
  cleanFilter.SetInputData(polydata)
  cleanFilter.Update()
  return cleanFilter.GetOutput()


def createPolyLineFromPoints(polydata):
  """Create a polyline from points.
  Args:
    polydata: vtkPolyData
  Returns:
    polyline: vtkPolyData
  """
  points = polydata.GetPoints()
  num_points = points.GetNumberOfPoints()
  
  lines = vtk.vtkCellArray()
  
  for i in range(num_points - 1):
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, i)
    line.GetPointIds().SetId(1, i + 1)
    lines.InsertNextCell(line)
  
  polyline = vtk.vtkPolyData()
  polyline.SetPoints(points)
  polyline.SetLines(lines)
  
  return polyline


def fitPolyLineToPoints(polydata):
  """
  Fit a polyline to points using PCA.
  Args:
    polydata: vtkPolyData
  Returns:
    polyline: vtkPolyData
  """

  points = polydata.GetPoints()
  num_points = points.GetNumberOfPoints()
  
  # Extract points into a numpy array
  points_array = np.zeros((num_points, 3))
  for i in range(num_points):
    points_array[i, :] = points.GetPoint(i)
  
  # Perform PCA to find the principal component
  pca = PCA(n_components=1)
  pca.fit(points_array)
  principal_component = pca.components_[0]
  
  # Project points onto the principal component
  projections = points_array @ principal_component
  
  # Sort points by their projection values
  sorted_indices = np.argsort(projections)
  sorted_points = vtk.vtkPoints()
  for i in sorted_indices:
    sorted_points.InsertNextPoint(points_array[i, :])
  
  # Create polyline from sorted points
  lines = vtk.vtkCellArray()
  for i in range(num_points - 1):
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, i)
    line.GetPointIds().SetId(1, i + 1)
    lines.InsertNextCell(line)
  
  polyline = vtk.vtkPolyData()
  polyline.SetPoints(sorted_points)
  polyline.SetLines(lines)
  
  return polyline

def computePCA(polydata, nComponent = 3):
  """
  Compute PCA of polydata.
  Args:
    polydata: vtkPolyData
    nComponent: int
  Returns:
    pca: PCA
  """
  vtk_points = polydata.GetPoints()
  points = np.array([vtk_points.GetPoint(i) for i in range(polydata.GetNumberOfPoints())])
  
  pca = PCA(n_components=nComponent)
  pca.fit(points)
  
  return pca


def createVectorPolyData(origin, direction, length=1):
  """
  Create a vector polydata from origin and vector.
  Args:
    origin: tuple
    vector: tuple
  Returns:
    polydata: vtkPolyData
  """

  # Create a line source
  line = vtk.vtkLineSource()
  line.SetPoint1(origin)
  line.SetPoint2([origin[i] + direction[i] * length for i in range(3)])
  line.Update()
  
  return line.GetOutput()

def createLabelPointData(polydata, value):
  """
  Create a label point data for polydata.
  Args:
    polydata: vtkPolyData
    value: int
  Returns:
    polydata: vtkPolyData
  """
  point_data = vtk.vtkIntArray()
  point_data.SetNumberOfComponents(1)
  point_data.SetName('Label')
  point_data.SetNumberOfTuples(polydata.GetNumberOfPoints())
  point_data.Fill(int(value))
  
  polydata.GetPointData().AddArray(point_data)
  
  return polydata

def assembleLabelMesh(polydataMap):
  """
  Assemble a label mesh from a map of label meshes.
  Args:
    polydataMap: dict (label -> vtkPolyData)
  Returns:
    polydata: vtkPolyData
  """
  appendFilter = vtk.vtkAppendPolyData()
  for label, polydata in polydataMap.items():
    polydata = createLabelPointData(polydata, label)
    appendFilter.AddInputData(polydata)
  appendFilter.Update()
  
  return appendFilter.GetOutput()


def createSquarePlane(origin, normal, length=100):
    """
    Create a square plane from origin and normal.
    Args:
        origin: tuple
        normal: tuple
        length: float
    Returns:
        polydata: vtkPolyData
    """
    # Normalize the normal vector
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)

    # Create two vectors perpendicular to the normal
    if np.allclose(normal, [0, 0, 1]):
        v1 = np.array([1, 0, 0])
    else:
        v1 = np.cross(normal, [0, 0, 1])
        v1 = v1 / np.linalg.norm(v1)
    
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)

    # Scale the vectors by the desired length
    v1 = v1 * length / 2
    v2 = v2 * length / 2

    # Define the four corners of the plane
    p0 = np.array(origin) - v1 - v2
    p1 = np.array(origin) + v1 - v2
    p2 = np.array(origin) + v1 + v2
    p3 = np.array(origin) - v1 + v2

    # Create a plane source
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(p0)
    plane.SetPoint1(p1)
    plane.SetPoint2(p3)
    plane.SetNormal(normal)
    plane.Update()

    return plane.GetOutput()


def createAngleFromPoints(origin, p1, p2):
  """
  Create an angle polydata from origin to p1 and p2.
  Args:
    origin: tuple
    p1: tuple
    p2: tuple
  Returns:
    polydata: vtkPolyData
  """
  # Create vtkPoints object
  points = vtk.vtkPoints()
  points.InsertNextPoint(origin)
  points.InsertNextPoint(p1)
  points.InsertNextPoint(p2)

  # Create lines
  lines = vtk.vtkCellArray()

  # First line (between pts[0] and pts[1])
  line1 = vtk.vtkPolyLine()
  line1.GetPointIds().SetNumberOfIds(2)
  line1.GetPointIds().SetId(0, 0)
  line1.GetPointIds().SetId(1, 1)
  lines.InsertNextCell(line1)

  # Second line (between pts[0] and pts[2])
  line2 = vtk.vtkPolyLine()
  line2.GetPointIds().SetNumberOfIds(2)
  line2.GetPointIds().SetId(0, 0)
  line2.GetPointIds().SetId(1, 2)
  lines.InsertNextCell(line2)

  # Create a polydata object
  polydata = vtk.vtkPolyData()
  polydata.SetPoints(points)
  polydata.SetLines(lines)

  return polydata


def vtk_to_sitk(vtk_image):
    # Extract dimensions, spacing, and origin
    dims = vtk_image.GetDimensions()
    spacing = vtk_image.GetSpacing()
    origin = vtk_image.GetOrigin()

    # Extract the scalar data as a NumPy array
    scalars = vtk_image.GetPointData().GetScalars()
    np_array = numpy_support.vtk_to_numpy(scalars)
    np_array = np_array.reshape(dims[2], dims[1], dims[0])  # Reshape to (z, y, x)

    # Create a SimpleITK image
    sitk_image = sitk.GetImageFromArray(np_array)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)

    # Set the direction (identity matrix for default)
    direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    sitk_image.SetDirection(direction)

def polydata_to_image(polydata, spacing=(1.0, 1.0, 1.0), dimensions=(100, 100, 100)):
    # Get bounds of the mesh (xmin, xmax, ymin, ymax, zmin, zmax)
    bounds = polydata.GetBounds()
    print(f"Mesh bounds: {bounds}")

    # Define image volume parameters
    origin = (bounds[0], bounds[2], bounds[4])  # start of volume grid
    dims = dimensions
    spacing = spacing

    # Create an empty vtkImageData volume
    image = vtk.vtkImageData()
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDimensions(dims)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # Initialize the volume with 1
    extent = image.GetExtent()
    for z in range(extent[4], extent[5]+1):
        for y in range(extent[2], extent[3]+1):
            for x in range(extent[0], extent[1]+1):
                image.SetScalarComponentFromFloat(x, y, z, 0, 1)

    # Convert mesh to stencil
    poly_to_stencil = vtk.vtkPolyDataToImageStencil()
    poly_to_stencil.SetInputData(polydata)
    poly_to_stencil.SetOutputOrigin(origin)
    poly_to_stencil.SetOutputSpacing(spacing)
    poly_to_stencil.SetOutputWholeExtent(image.GetExtent())
    poly_to_stencil.Update()

    # Apply stencil to image to fill inside mesh voxels with 1
    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(image)
    stencil.SetStencilData(poly_to_stencil.GetOutput())
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(0)
    stencil.Update()

    # Extract vtkImageData from stencil filter
    output_image = stencil.GetOutput()

    # Convert vtkImageData to simpleITK image
    return vtk_to_sitk(output_image)