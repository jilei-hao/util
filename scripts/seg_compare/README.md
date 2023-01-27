## PREREQUISITE
- Download c3d from http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.C3D or build one from the source code
- Download cmrep from https://sourceforge.net/projects/cmrep/files/cmrep/Nightly/ or build one from the source code
- Create link in a terminal using `ln [source] [target]` to make sure c3d, vtklevelset, and meshdiff can be directly executed from the terminal. See following example:
  - Run `echo $PATH` to make sure `/usr/local/bin` is in the list
  - Assuming c3d binary is in /some/path/to/c3d
  - Run `ln -s /some/path/to/c3d /usr/local/bin/c3d`
  - You should be able to directly run c3d just using `c3d [option] ...`
  - Do the same for vtkleveset and meshdiff

## USAGE
- Create a working directory, <strong>all the intermediary files will be stored here</strong>
- Organize input files into 2 folders. The file pair to be compared should have same filename. For example:
  - Tracer1
    - img1.nii.gz
    - img2.nii.gz
  - Tracer2
    - img1.nii.gz
    - img2.nii.gz
- It's users's responsibility to make sure each pair of image is comparable, or strange result could be generated by the script.
- Execute the script and redirect result to a log file, replace the path placeholders with real paths
  - `bash path/to/script/seg_compare.sh /input/dir/1 /input/dir/2 /work/dir > /path/to/your/result.log 2>&1`
  - The result and errors will be saved to the file `/path/to/your/result.log`
  - Without `> /path/to/your/result.log 2>&1` part, result will be printed in terminal