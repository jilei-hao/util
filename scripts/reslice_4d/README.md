# Reslice 4D Images Script
## PREREQUISITE
- Download c3d from http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.C3D or build one from the source code
- Create link in a terminal using `ln [source] [target]` to make sure c3d and c4d can be directly executed from the terminal. See following example:
  - Run `echo $PATH` to make sure `/usr/local/bin` is in the list
  - Assuming c3d binary is in /some/path/to/c3d
  - Run `ln -s /some/path/to/c3d /usr/local/bin/c3d`
  - You should be able to directly run c3d just using `c3d [option] ...`
  - Do the same for c4d
  - Create a working directory, <strong>all the intermediary files will be stored here</strong>

## USAGE
```bash
bash reslice_4d.sh <fixed_image> <moving_image> <reslice_command> <working_directory>
```
Output will be written with filename: `mov4d_resliced.nii.gz`

## EXAMPLE
Example using `-reslice-identity`
```bash
bash reslice_4d.sh fix.nii.gz mov.nii.gz -reslice-identity /work
```
Example using `-reslice-itk <transform_matrix>`
```bash
bash reslice_4d.sh fix.nii.gz mov.nii.gz -reslice-itk /path/to/transform.txt /work
```
Example using `-reslice-matrix <transform_matrix>`
```bash
bash reslice_4d.sh fix.nii.gz mov.nii.gz -reslice-matrix /path/to/transform.txt /work
```