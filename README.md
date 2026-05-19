# SolarSimulatorPython

## HPC login

`ssh <username>@login.genome.au.dk`

## Queue batch jobs

Replace \<offset\> with your own unique id.

`cd DeiC-RUC-L2-202601/SolarSimulatorPython`

`sbatch batch_run.sh 10000 50000 1 <offset>`

This will give you a job id which you can use to check the job progress:

`jobinfo <job-id>`

## Retrieve data

Linux/WSL/MacOS only.

Replace \<username\>.

### All data

`rsync <username>@login.genome.au.dk:/faststorage/project/DeiC-RUC-L2-202601/SolarSimulatorPython/data ./`

### Data with specific id

`rsync -av --include="*_<id>.*" --exclude="*" <username>@login.genome.au.dk:/faststorage/project/DeiC-RUC-L2-202601/SolarSimulatorPython/data/ ./`

Replace \<id\> with the id of the dataset.

### Specific file

`rsync <username>@login.genome.au.dk:/faststorage/project/DeiC-RUC-L2-202601/SolarSimulatorPython/data/<filename> ./`

Replace \<filename\>.
