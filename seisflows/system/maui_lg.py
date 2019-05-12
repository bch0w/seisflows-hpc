
import os
import math
import sys
import time

from os.path import abspath, basename, exists, join
from subprocess import check_output, call
from uuid import uuid4
from seisflows.tools import unix
from seisflows.tools.tools import call, findpath
from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class maui_lg(custom_import('system', 'slurm_lg')):
    """ System interface for NeSI HPC maui and ancillary node maui_ancil

      For more informations, see 
      http://seisflows.readthedocs.org/en/latest/manual/manual.html#system-interfaces
    """

    def check(self):
        """ Checks parameters and paths
        """
        # limit on number of concurrent tasks
        if 'NTASKMAX' not in PAR:
            setattr(PAR, 'NTASKMAX', 100)

        # number of cores per node
        if 'NODESIZE' not in PAR:
            setattr(PAR, 'NODESIZE', 24)

        # how to invoke executables
        if 'MPIEXEC' not in PAR:
            setattr(PAR, 'MPIEXEC', 'srun')

        # optional additional SLURM arguments
        if 'SLURMARGS' not in PAR:
            setattr(PAR, 'SLURMARGS', '--partition=t1small')

        # optional environment variable list VAR1=val1,VAR2=val2,...
        if 'ENVIRONS' not in PAR:
            setattr(PAR, 'ENVIRONS', '')

        # where temporary files are written, (overwrite scratch in paths.py)
        if 'SCRATCH' not in PATH:
            setattr(PATH, 'SCRATCH', join(os.getenv('CENTER1'), 'scratch', str(uuid4())))

        super(maui_lg, self).check()

    def submit(self, workflow):
        """ Submits workflow to maui_ancil cluster
        This needs to be run on maui_ancil because maui does not have the 
        ability to run the command "sacct"
        """
        # create scratch directories
        unix.mkdir(PATH.SCRATCH)
        unix.mkdir(PATH.SYSTEM)

        # create output directories
        unix.mkdir(PATH.OUTPUT)
        unix.mkdir(PATH.WORKDIR+'/'+'output.slurm')
        if not exists('./scratch'): 
            unix.ln(PATH.SCRATCH, PATH.WORKDIR+'/'+'scratch')

        workflow.checkpoint()
        
        # Submit to maui_ancil
        call('sbatch '
                + '%s ' % PAR.SLURMARGS
                + '--clusters=%s ' % 'maui_ancil'
                + '--partition=%s ' % 'nesi_prepost'
                + '--job-name=%s ' % PAR.TITLE
                + '--output=%s ' % (PATH.WORKDIR+'/'+'output.log')
                + '--error=%s ' % (PATH.WORKDIR+'/'+'error.log')
                + '--tasks=%d ' % 1 # PAR.NODESIZE
                + '--cpus-per-task=%d ' % 1
                + '--time=%d ' % PAR.WALLTIME
                + findpath('seisflows.system') +'/'+ 'wrappers/submit '
                + PATH.OUTPUT)

    def run(self, classname, method, *args, **kwargs):
        """ Runs task multiple times in embarrassingly parallel fasion on the
            maui cluster

          Executes classname.method(*args, **kwargs) NTASK times, each time on
          NPROC cpu cores
        """
        self.checkpoint(PATH.OUTPUT, classname, method, args, kwargs)
        
        stdout = check_output(
                   'sbatch %s ' % PAR.SLURMARGS
                   + '--job-name=%s ' % PAR.TITLE
                   + '--clusters=%s ' % 'maui'
                   + '--partition=%s ' % 'nesi_research'
                   + '--cpus-per-task=%s ' % 1
                   + '--nodes=%d ' % 2
                   + '--ntasks=%d ' % PAR.NPROC
                   + '--time=%d ' % PAR.TASKTIME
                   + '--output %s ' % (PATH.WORKDIR+'/'+'output.slurm/'+'%A_%a')
                   + '--array=%d-%d ' % (0, (PAR.NTASK-1) % PAR.NTASKMAX)
                   + '%s ' % (findpath('seisflows.system') +'/'+ 'wrappers/run')
                   + '%s ' % PATH.OUTPUT
                   + '%s ' % classname
                   + '%s ' % method
                   + '%s ' % PAR.ENVIRONS,
                   shell=True)

        # keep track of job ids
        jobs = self.job_id_list(stdout, PAR.NTASK)

        # check job array completion status
        while True:
            # wait a few seconds between queries
            time.sleep(5)

            isdone, jobs = self.job_array_status(classname, method, jobs)
            if isdone:
                return

    def run_preproc(self, classname, method, *args, **kwargs):
        """ Runs task a single time. For Maui this is run on maui ancil
        and also includes some extra arguments for eval_func
        """
        self.checkpoint(PATH.OUTPUT, classname, method, args, kwargs)

        # submit job
        stdout = check_output(
                   'sbatch %s ' % PAR.SLURMARGS
                   + '--job-name=%s ' % PAR.TITLE
                   + '--tasks=%d ' % 1
                   + '--cpus-per-task=%d' % 1
                   + '--account=%s' % 'nesi00263'
                   + '--partition=n%s' % 'nesi_prepost'
                   + '--time=%d ' % 15  # ancil preprocessing is short
                   + '--array=%d-%d ' % (0, (PAR.NTASK - 1) % PAR.NTASKMAX)
                   + '--output %s ' % (PATH.WORKDIR+'/'+'output.slurm/'+'%A_%a')
                   + '%s ' % (findpath('seisflows.system') +'/'+ 'wrappers/run')
                   + '%s ' % PATH.OUTPUT
                   + '%s ' % classname
                   + '%s ' % method
                   + '%s ' % PAR.ENVIRONS,
                   shell=True)

        # keep track of job ids
        jobs = self.job_id_list(stdout, 1)

        # check job completion status
        while True:
            # wait a few seconds between queries
            time.sleep(5)

            isdone, jobs = self.job_array_status(classname, method, jobs)
            if isdone:
                return

    def job_status(self, job):
        """ Queries completion status of a single job
            added a -L flag to sacct to query all clusters
        """
        stdout = check_output(
            'sacct -nL -o jobid,state -j ' + job.split('_')[0],
            shell=True)

        state = ''
        lines = stdout.strip().split('\n')
        for line in lines:
            if line.split()[0] == job:
                state = line.split()[1]
        return state

    def job_id_list(self, stdout, ntask):
        """Parses job id list from sbatch standard output
        
        Modified because submitting jobs across clusters on maui means the 
        phrase "on cluster X" gets appended to the end of stdout and the job #
        is no longer stdout().split()[-1]

        Instead, scan through stdout and try to find the number using float() to
        break on words
        """    
        for parts in stdout.split():
            try:
                job_id = parts.strip() 
                test_break = float(job_id)
                return [job_id+str(ii) for ii in range(ntask)]
            except ValueError:
                continue
    

