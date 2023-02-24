# Run Batch Python Propagation on PMACS Cluster
## Pre-requisites
### PMACS Configuration
1. Create a PMACS account. ([PMACS Help](https://www.med.upenn.edu/dart/need-help/))
2. Add your account to the `picsl` group
3. Follow the documentation for setting up VPN and connecting to the LPC
### PMACS Documentations
For information about connecting to and accessing the LPC, please see our wiki page (PennKey Authentication).
https://wiki.pmacs.upenn.edu/pub/LPC

If you are not familiar with our scheduling software, this section may be helpful.
https://wiki.pmacs.upenn.edu/pub/LSF_Basics
https://wiki.pmacs.upenn.edu/pub/Batch_Computing

Applications are installed on a central shared directory so the applications are available to all the hosts in our cluster. "modules" can be used to see what is available and to load or unload software packages.
https://wiki.pmacs.upenn.edu/pub/LPC#Modules

VPN download and configure
https://www.med.upenn.edu/dart/vpn-instructions.html

Configuring 2 factor authentication for VPN.
https://www.isc.upenn.edu/how-to/two-step-verification-getting-started
https://www.med.upenn.edu/dart/pulse-vpn-duo-instructions.html

## Share Data
Once added to the picsl group, you will have access to the `/project/picsl` folder. Create a user directory here to store and share data with other picsl users. <span style="color:red">To make the shared directory clean and organized,please always create a user folder. Don't store data directly under the picsl directory. </span>

After creating the user directory, use
```bash
chgrp -R picsl your_directory
```
to change the group ownership of your folder, so other picsl memebers can access your data

You can store any data not to be shared at `/home/your_user_name/`

Transferring data between local and remote: https://linuxize.com/post/how-to-use-scp-command-to-securely-transfer-files/

## Run Propagation Batch Script on the cluster
For documentation running batch script on any computers, see: https://github.com/jilei-hao/segmentation-propagation/blob/main/README.md

Example batch script and the config file are located at `/project/picsl/jhao/propagation/example_run`. The `cluster_config.json` stores links to the tool binary compiled on the machine. You can use this file to configure your own run, or create another file to customize some of the parameters. The input and output of the example run are stored at `/project/picsl/jhao/propagation/example`

It is recommended to always test your configuration using small data sample and limited target timepoints using the interactive shell `ibash` before submitting the job formally. 

It is also recommended to create your own copies of `cluster_config.json` and `example_bsub_batch_run.sh` (you can rename them), instead of using the ones in the example_run folder. Because the shared example could be used and modified by other users.

### Example: Create you own run
1. Use scp or other tools to upload your data to the cluster and store them at `/your/data/folder`
2. Copy the `cluster_config.json` and `example_bsub_batch_run.sh` to `/your/script/folder`. Modify them to use your own settings, emails, filelist, etc. 
3. (Optional) Execute a testing run using the interactive shell `ibash`.
4. (Optional) Exit the interactive shell using command `exit`
5. Submit the job using the `bsub` command. e.g. `bsub -n 1,32 < your_run_script.sh`