
import os
import math
import sys
import time

from os.path import abspath, basename, exists, join
from subprocess import check_output, call, CalledProcessError
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
            setattr(PAR, 'NODESIZE', 40)

        # how to invoke executables
        if 'MPIEXEC' not in PAR:
            setattr(PAR, 'MPIEXEC', 'srun')

        # optional environment variable list VAR1=val1,VAR2=val2,...
        if 'ENVIRONS' not in PAR:
            setattr(PAR, 'ENVIRONS', '')
        
        # NeSI cluster Maui and Maui Anciliary Node specific variables
        if 'ACCOUNT' not in PAR:
            raise Exception()        
    
        if 'MAIN_CLUSTER' not in PAR:
            setattr(PAR, 'MAIN_CLUSTER', 'maui')
              
        if 'MAIN_PARTITION' not in PAR:
            setattr(PAR, 'MAIN_PARTITION', 'nesi_research')
    
        if 'ANCIL_CLUSTER' not in PAR:
            setattr(PAR, 'ANCIL_CLUSTER', 'maui_ancil')
        
        if 'ANCIL_PARTITION' not in PAR:
            setattr(PAR, 'ANCIL_PARTITION', 'nesi_prepost')
        
        if 'ANCIL_TASKTIME' not in PAR:
            setattr(PAR, 'ANCIL_TASKTIME', PAR.TASKTIME)

        # if number of nodes not given, automatically calculate.
        # if the 'nomultithread' hint is given, the number of nodes will need 
        # to be manually set above this auto calculation         
        if 'NODES' not in PAR:
            setattr(PAR, 'NODES', math.ceil(PAR.NPROC/float(PAR.NODESIZE)))

        if 'CPUS_PER_TASK' not in PAR:
            setattr(PAR, 'CPUS_PER_TASK', 1)
        
        if 'WITH_OPENMP' not in PAR:
            setattr(PAR, 'WITH_OPENMP', False)
        if PAR.WITH_OPENMP:
        # Make sure that y * z = c * x; where x=nodes, y=ntasks= z=cpus-per-task
        # and c=number of cores per node, if using OpenMP
            assert(PAR.NPROC * PAR.CPUS_PER_TASK == PAR.NODESIZE * PAR.NODES)

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

        # if resuming, rename the old log files so they don't get overwritten
        output_log = os.path.join(PATH.WORKDIR, 'output.log')
        error_log = os.path.join(PATH.WORKDIR, 'error.log')
        for log in [output_log, error_log]:
            log_prior = log + '_prior'
            log_temp = log + '_temp'
            if os.path.exists(log):
                # If a prior log exists, move to temp file and then rewrite 
                # with new log file
                if os.path.exists(log_prior):
                    os.rename(log_prior, log_temp)
                    with open(log_prior, 'w') as f_out:
                        for fid in [log_temp, log]:
                            with open(fid) as f_in:
                                f_out.write(f_in.read())
                    unix.rm(log_temp)
                else:
                    os.rename(log, log_prior)
                    
        workflow.checkpoint()
               
        # Submit to maui_ancil 
        call(" ".join([
            'sbatch',
            '%s' % PAR.SLURMARGS,
            '--account=%s' % PAR.ACCOUNT,
            '--clusters=%s' % PAR.ANCIL_CLUSTER,
            '--partition=%s' % PAR.ANCIL_PARTITION,
            '--job-name=%s' %  'M_' + PAR.TITLE,
            '--output=%s' % output_log,
            '--error=%s' % error_log,
            '--ntasks=%d' % 1,
            '--cpus-per-task=%d' % 1,
            '--time=%d' % PAR.WALLTIME,
            findpath('seisflows.system') +'/'+ 'wrappers/submit ',
            PATH.OUTPUT])
            )

    def prep_openmp(self, subprocess_call):
        """
        OpenMP requires some initial exports before sbatch command can be run
        This function will append these to the 'check_output' call that 'run'
        and 'run_single' use

        :type subprocess_call: str
        :param subprocess_call: the string that is passed to check_output
        :rtype: str
        :return: a prepended call with the correct export statements
        """
        prepended_call = " ".join([
                    "export OMP_NUM_THREADS={};".format(PAR.CPUS_PER_TASK),
                    "export OMP_PROC_BIND=true;",
                    "export OMP_PLACES=cores;",
                    subprocess_call
                    ])
        
        return prepended_call

    def run(self, classname, method, scale_tasktime=1, *args, **kwargs):
        """ Runs task multiple times in embarrassingly parallel fasion on the
            maui cluster

          Executes classname.method(*args, **kwargs) NTASK times, each time on
          NPROC cpu cores
        """
        self.checkpoint(PATH.OUTPUT, classname, method, args, kwargs)
        run_call = " ".join([
            'sbatch',
            '%s' % PAR.SLURMARGS,
            '--account=%s' % PAR.ACCOUNT,
            '--job-name=%s' % PAR.TITLE,
            '--clusters=%s' % PAR.MAIN_CLUSTER,
            '--partition=%s' % PAR.MAIN_PARTITION,
            '--cpus-per-task=%s' % PAR.CPUS_PER_TASK,
            '--nodes=%d' % PAR.NODES,
            '--ntasks=%d' % PAR.NPROC,
            '--time=%d' % (PAR.TASKTIME * scale_tasktime),
            '--output %s' % os.path.join(PATH.WORKDIR, 'output.slurm/'+'%A_%a'),
            '--array=%d-%d' % (0, (PAR.NTASK-1) % PAR.NTASKMAX),
            '%s' % (findpath('seisflows.system') +'/'+ 'wrappers/run'),
            '%s' % PATH.OUTPUT,
            '%s' % classname,
            '%s' % method,
            '%s' % PAR.ENVIRONS])
        
        if PAR.WITH_OPENMP:
            run_call = self.prep_openmp(run_call) 

        stdout = check_output(run_call, shell=True)
        
        # keep track of job ids
        jobs = self.job_id_list(stdout, PAR.NTASK)

        # check job completion status
        check_status_error = 0
        while True:
            # wait a few seconds between queries
            time.sleep(5)
            # Occassionally connections using 'sacct' are refused leading to job
            # failure. Wrap in a try-except and allow a handful of failures 
            # incase the failure was a one-off connection problem
            try:
                isdone, jobs = self.job_array_status(classname, method, jobs)
            except CalledProcessError:
                check_status_error += 1
                if check_status_error >= 10:
                    print "check job status with sacct failed 10 times"
                    sys.exit(-1)
                pass
            if isdone:
                return
        
    def run_single(self, classname, method, scale_tasktime=1, *args, **kwargs):
        """ Runs task a single time

          Executes classname.method(*args, **kwargs) a single time on NPROC
          cpu cores
        """
        self.checkpoint(PATH.OUTPUT, classname, method, args, kwargs)

        # submit job
        run_call = " ".join([
            'sbatch', 
            '%s ' % PAR.SLURMARGS,
            '--account=%s' % PAR.ACCOUNT,
            '--job-name=%s' % PAR.TITLE,
            '--clusters=%s' % PAR.MAIN_CLUSTER,
            '--partition=%s' % PAR.MAIN_PARTITION,
            '--cpus-per-task=%s' % PAR.CPUS_PER_TASK,
            '--ntasks=%d' % PAR.NPROC,
            '--nodes=%d' % PAR.NODES,
            '--time=%d' % (PAR.TASKTIME * scale_tasktime),
            '--array=%d-%d' % (0,0),
            '--output %s' % (PATH.WORKDIR+'/'+'output.slurm/'+'%A_%a'),
            '%s' % (findpath('seisflows.system') +'/'+ 'wrappers/run'),
            '%s' % PATH.OUTPUT,
            '%s' % classname,
            '%s' % method,
            '%s' % PAR.ENVIRONS,
            '%s' % 'SEISFLOWS_TASKID=0'])

        if PAR.WITH_OPENMP:
            run_call = self.prep_openmp(run_call)
        
        stdout = check_output(run_call, shell=True)
        
        # keep track of job ids
        jobs = self.job_id_list(stdout, 1)

        # check job completion status
        check_status_error = 0
        while True:
            # wait a few seconds between queries
            time.sleep(5)
            # Occassionally connections using 'sacct' are refused leading to job
            # failure. Wrap in a try-except and allow a handful of failures 
            # incase the failure was a one-off connection problem
            try:
                isdone, jobs = self.job_array_status(classname, method, jobs)
            except CalledProcessError:
                check_status_error += 1
                if check_status_error >= 10:
                    print "check job status with sacct failed 10 times"
                    sys.exit(-1)
                pass
            if isdone:
                return

    def run_ancil(self, classname, method, *args, **kwargs):
        """ Runs task a single time. For Maui this is run on maui ancil
        and also includes some extra arguments for eval_func
        """
        self.checkpoint(PATH.OUTPUT, classname, method, args, kwargs)

        # submit job
        stdout = check_output(" ".join([
            'sbatch', 
            '%s' % PAR.SLURMARGS,
            '--job-name=%s' % PAR.TITLE,
            '--tasks=%d' % 1,
            '--cpus-per-task=%d' % PAR.CPUS_PER_TASK,
            '--account=%s' % PAR.ACCOUNT,
            '--clusters=%s' % PAR.ANCIL_CLUSTER,
            '--partition=%s' % PAR.ANCIL_PARTITION,
            '--time=%d' % PAR.ANCIL_TASKTIME,
            '--array=%d-%d' % (0, (PAR.NTASK - 1) % PAR.NTASKMAX),
            '--output %s' % (PATH.WORKDIR+'/'+'output.slurm/'+'%A_%a'),
            '%s' % (findpath('seisflows.system') +'/'+ 'wrappers/run'),
            '%s' % PATH.OUTPUT,
            '%s' % classname,
            '%s' % method,
            '%s' % PAR.ENVIRONS]),
            shell=True
            )

        # keep track of job ids
        jobs = self.job_id_list(stdout, 1)

        # check job completion status
        check_status_error = 0
        while True:
            # wait a few seconds between queries
            time.sleep(5)
            # Occassionally connections using 'sacct' are refused leading to job
            # failure. Wrap in a try-except and allow a handful of failures 
            # incase the failure was a one-off connection problem
            try:
                isdone, jobs = self.job_array_status(classname, method, jobs)
            except CalledProcessError:
                check_status_error += 1
                if check_status_error >= 10:
                    print "check job status with sacct failed 10 times"
                    sys.exit(-1)
                pass
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
                return [job_id+'_'+str(ii) for ii in range(ntask)]
            except ValueError:
                continue
    

