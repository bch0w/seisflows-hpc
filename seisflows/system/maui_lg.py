
import os
import math
import sys
import time

from os.path import abspath, basename, exists, join
from subprocess import check_output
from uuid import uuid4
from seisflows.tools import unix
from seisflows.tools.tools import call, findpath
from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class maui_lg(custom_import('system', 'slurm_lg')):
    """ System interface for University of Alaska Fairbanks CHINOOK

      If you are using more than 48 cores per task, then add the following to
      your parameter file:
          SLURMARGS='--partition=t1standard'      

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

        # where temporary files are written
        if 'SCRATCH' not in PATH:
            setattr(PATH, 'SCRATCH', join(os.getenv('CENTER1'), 'scratch', str(uuid4())))

        super(maui_lg, self).check()


    def submit(self, workflow):
        """ Submits workflow to maui_ancil cluster
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
        # prepare sbatch arguments
        import pdb;pdb.set_trace()
        call('sbatch '
                + '%s ' % PAR.SLURMARGS
                + '--clusters=%s' % 'maui_ancil'
                + '--partition=%s ' % 'nesi_prepost'
                + '--hint=%s ' % 'nomultithread'
                + '--job-name=%s ' % PAR.TITLE
                + '--output %s ' % (PATH.WORKDIR+'/'+'output.log')
                + '--ntasks=%d ' % PAR.NODESIZE
                + '--nodes=%d ' % 1
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

        # submit job array
        stdout = check_output(
                   'sbatch %s ' % PAR.SLURMARGS
                   + '--clusters=%s' % 'maui'
                   + '--partition=%s ' % 'nesi_research'
                   + '--job-name=%s ' % PAR.TITLE
                   + '--nodes=%d ' % math.ceil(PAR.NPROC/float(PAR.NODESIZE))
                   + '--ntasks-per-node=%d ' % PAR.NODESIZE
                   + '--ntasks=%d ' % PAR.NPROC
                   + '--time=%d ' % PAR.TASKTIME
                   + '--array=%d-%d ' % (0,(PAR.NTASK-1)%PAR.NTASKMAX)
                   + '--output %s ' % (PATH.WORKDIR+'/'+'output.slurm/'+'%A_%a')
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

    def mpiexec(self):
        """ Specifies MPI exectuable; used to invoke solver
        """
        return 'mpirun -np %d ' % PAR.NPROC

