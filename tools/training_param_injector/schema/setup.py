'''To generate circle_schema_generated.py and traininfo_generated.py'''

import wget
import zipfile
import os.path 
import stat
import subprocess

# unzip flatc23.5.6
# generate schema

FLATC_V23_5_26 = 'https://github.com/google/flatbuffers/releases/download/v23.5.26/Linux.flatc.binary.g++-10.zip'
FLATC_ZIP = 'flatc.zip'
FLATC_EXE = 'flatc'
 
ONE_
CIRCLE_SCHEMA = '../../'
TINFO_SCHEMA = '..'

class FlatcCaller:

  def __call__(self, schema_paths):
    self.__download_flatc()
    for schema_path in schema_paths: 
      self.__generate_python_schema(schema_path)
    self.__clear_flatc()


  def __download_flatc(self):
    '''Download flatc and unip it in current directory'''
    wget.download(FLATC_V23_5_26, FLATC_ZIP)

    # unzip 'flatc' in current directory    
    with zipfile.ZipFile(FLATC_ZIP, 'r') as zip:
      for f in zip.infolist():
        print(f.filename)
        if f.filename == FLATC_EXE :
          zip.extract(f, '.')

    if not os.path.isfile(FLATC_EXE):
      raise RuntimeError('Failed to download flatc')
    
    # add permission to execute
    perm = os.stat(FLATC_EXE)
    os.chmod(FLATC_EXE, perm.st_mode | stat.S_IXUSR)


  def __generate_python_schema(self, schema_path, out_dir ='.'):
    '''execute flatc to compile *.fbs into python file'''    
    if not os.path.isfile(FLATC_EXE):
      raise RuntimeError('Failed to find flatc')
    if not os.stat(FLATC_EXE).st_mode | stat.S_IXUSR:
      raise RuntimeError('No permission to execute flatc')
    
    cmd = [FLATC_EXE, '--python', '--gen_object-api', 'o', out_dir, schema_path]
    try:
      subprocess.check_call(cmd)
    except Exception as e: 
      print("failed to compile using flatc", e)
 

  def __clear_flatc(self):
    if os.path.exists(FLATC_ZIP) : 
      os.remove(FLATC_ZIP)
    if os.path.exists(FLATC_EXE) : 
      os.remove(FLATC_EXE)


# init this directory
run_flatc = FlatcCaller()
run_flatc([CIRCLE_SCHEMA, TINFO_SCHEMA])

