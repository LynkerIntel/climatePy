import os
import platform
import re
import warnings

def whatOS():
	os_name = platform.system().lower()
	if os_name == 'darwin':
		os_name = 'osx'
	return os_name

# build_file = function(file){
#   if (whatOS() == "windows") {
#     paste0(Sys.getenv("UserProfile"), "\\", paste0(".", file))
#   } else {
#     paste0(Sys.getenv("HOME"), "/" , paste0(".", file))
#   }
# }

def build_file(file):
	if whatOS() == "windows":
		return os.path.join(os.environ['UserProfile'], "." + file)
	else:
		return os.path.join(os.environ['HOME'], "." + file)
	
def getDodsrcPath():

	"""Get a default dodsrc file path
	
	Args: None

	Returns: A string containing the default netrc file path
	"""

	return build_file("dodsrc")


def getNetrcPath():
	
		"""Get a default netrc file path
		
		Args: None
	
		Returns: A string containing the default netrc file path
		"""
	
		return build_file("netrc")

def writeNetrc(login,
			password, 
			machine = 'urs.earthdata.nasa.gov', 
			netrcFile = getNetrcPath(),
			overwrite = False
			):
		
		"""Write netrc file
		
		Write a netrc file that is valid for accessing urs.earthdata.nasa.gov

		Args:
			login (str): Email address used for logging in on earthdata
			password (str): Password associated with the login.
			machine (str): the machine you are logging into
			netrcFile (str): A path to where the netrc file should be written.
				By default will go to your home directory, which is advised
			overwrite (bool): overwrite the existing netrc file?
		
		Returns:
			A character vector containing the netrc file path
		"""
		
		if (login is None) or (password is None):
			raise ValueError("Login/Password is missing. If you dont have an account please registar at:\nhttps://urs.earthdata.nasa.gov/users/new")
		
		if os.path.exists(netrcFile) and not overwrite:
			raise ValueError(f"{netrcFile} already exists. Set `overwrite=True` if you'd like to overwrite.")
		
		string = f"\nmachine {machine}\n\
			login {login}\n\
			password {password}"
		
		# create a netrc file
		with open(os.path.expanduser(netrcFile), 'w') as f:
			f.write(string)
			
		# set the owner-only permission
		os.chmod(netrcFile, 0o600)
		
		return netrcFile

def checkNetrc(netrcFile = getNetrcPath(), machine = "urs.earthdata.nasa.gov" ):

	"""Check netrc file

	Check that there is a netrc file with a valid entry for urs.earthdata.nasa.gov.

	Args:
		netrcFile (str): File path to netrc file to check.
		machine (str): the machine you are logging into
	
	Returns:
		boolean, If True, there is a valid entry for urs.earthdata.nasa.gov in the netrc file, otherwise False.

	"""

	# check if netrc file exists
	if not os.path.exists(netrcFile):
		return False
	
	# read lines from netrc file
	with open(netrcFile, 'r') as f:
		lines = f.readlines()

	# remove http:// or https:// from lines
	lines = [re.sub("http.*//", "", line) for line in lines]

	# check if 'machine' is any of the lines, if so return True, otherwise False
	return any([re.search(machine, line) for line in lines])

def writeDodsrc(netrcFile = getNetrcPath(), dodsrcFile = "./dodsrc"):

	"""Write dodsrc file
	Write a dodsrc file that is valid for a netrc file

	Args:
		netrcFile (str): A path to where the netrc file should be.
		dodsrcFile (str): The path to the dodsrc file you want to write.
			By default will go to your home directory, which is advised

	Returns:
		A string containing the netrc file path
	"""


	# netrcFile = getNetrcPath()
	# dodsrcFile = "./dodsrc"
	# # dodsrcFile =getDodsrcPath()

	if os.path.exists(dodsrcFile):
		os.unlink(dodsrcFile)

	dir = os.path.dirname(dodsrcFile)
	# dir = os.path.realpath(dodsrcFile)
	
	# build the dodsrc file string
	string = (
		f"USE_CACHE=0\n"
		f"MAX_CACHE_SIZE=20\n"
		f"MAX_CACHED_OBJ=5\n"
		f"IGNORE_EXPIRES=0\n"
		f"DEFAULT_EXPIRES=86400\n"
		f"ALWAYS_VALIDATE=0\n"
		f"DEFLATE=0\n"
		f"VALIDATE_SSL=1\n"
		f"HTTP.COOKIEJAR={dir}/.urs_cookies\n"
		f"HTTP.NETRC={netrcFile}"
		)
	
	# create a netrc file
	with open(dodsrcFile, 'w') as f:
		f.write(string)
	
	# set the owner-only permission
	os.chmod(dodsrcFile, 0o600)

	return dodsrcFile


def checkDodsrc(dodsrcFile = getDodsrcPath(), netrcFile = getNetrcPath()):
# def checkDodsrc(dodsrcFile = getDodsrcPath(), netrcFile = None):
	"""Check dodsrc file

	Check that there is a dodsrc file with a valid entry for urs.earthdata.nasa.gov.

	Args:
		dodsrcFile (str): File path to dodsrc file to check. Default is path generated from getDodsrcPath().
		netrcFile (str): File path to netrc file to check. Default is path generated from getNetrcPath().

	Returns:
		boolean, If True, there is a valid entry for urs.earthdata.nasa.gov in the dodsrc file, otherwise False.
	"""

	# dodsrcFile = getDodsrcPath()
	# if netrcFile is None:
	# 	netrcFile = getNetrcPath()

	# check if netrc file exists
	if not os.path.exists(netrcFile):
		return False
	
	# check if dodsrc file exists
	if not os.path.exists(dodsrcFile):
		return False
	
	# read lines from dodsrc file
	with open(dodsrcFile, 'r') as f:
		lines = f.readlines()
	
	# remove http:// or https:// from lines
	lines = [re.sub("http.*//", "", line) for line in lines]
	
	# check if 'machine' is any of the lines, if so return True, otherwise False
	return any([re.search(netrcFile, line) for line in lines])

def check_rc_files(dodsrcFile = getDodsrcPath(), netrcFile = getNetrcPath()):

	"""Check rc files

	Check that there is a dodsrc file with a valid entry for urs.earthdata.nasa.gov.
	If not, check that there is a netrc file with a valid entry for urs.earthdata.nasa.gov.
	If so, write a dodsrc file with the path to the netrc file.

	Args:
		dodsrcFile (str): File path to dodsrc file to check. Default is path generated from getDodsrcPath().
		netrcFile (str): File path to netrc file to check. Default is path generated from getNetrcPath().

	Returns:
		None
	"""

	if not checkDodsrc(dodsrcFile, netrcFile):
		if checkNetrc(netrcFile):
			print(f"Found Netrc file. Writing dodsrs file to: {getDodsrcPath()}")
			# warnings.warn(f"Found Netrc file. Writing dodsrs file to: {getDodsrcPath()}")
			writeDodsrc(netrcFile, dodsrcFile)
		else:
			# raise Exception("Netrc file not found. Please run writeNetrc() with earth data credentials.")
			print("Netrc file not found. Please run writeNetrc() with earth data credentials.")
			
# ###########

# import os
# import platform
# import re

# def whatOS():
#     os_name = platform.system().lower()
#     if os_name == 'darwin':
#         os_name = 'osx'
#     return os_name

# def getNetrcPath():
#     home = os.environ['HOME']
#     if whatOS() == "windows":
#         return os.path.join(home, "_netrc")
#     else:
#         return os.path.join(home, ".netrc")

# def getDodsrcPath():
#     home = os.getenv("HOME")
#     if whatOS() == "windows":
#         return os.path.join(home, '_dodsrc')
#     else:
#         return os.path.join(home, '.dodsrc')

# def writeNetrc(
#         login     = None, 
#         password  = None, 
#         machine   = 'urs.earthdata.nasa.gov',
#         netrcFile = None, 
#         overwrite = False
#         ):
    
#     """Write netrc file"""

#     if not netrcFile:
#         netrcFile = getNetrcPath()
    
#     if login is None or password is None:
#         raise ValueError("Login/Password is missing. If you don't have an account please register at:\nhttps://urs.earthdata.nasa.gov/users/new")
    
#     if os.path.exists(netrcFile) and not overwrite:
#         raise FileExistsError("'" + netrcFile + "' already exists. Set `overwrite=True` if you'd like to overwrite.")
    
#     string = f"\nmachine {machine}\nlogin {login}\npassword {password}"
    
#     with open(os.path.expanduser(netrcFile), 'a') as f:
#         f.write(string)
    
#     os.chmod(netrcFile, 0o600)
    
#     return netrcFile

# def checkNetrc(
#         netrcFile = getNetrcPath(),
#         machine   = "urs.earthdata.nasa.gov"
#         ):
#     """Check netrc file"""

#     if not os.path.exists(netrcFile):
#         return False
    
#     with open(netrcFile) as f:
#         lines = f.readlines()

#     lines = [line.replace("http.*//", "") for line in lines]

#     return any(machine in line for line in lines)

# def checkDodsrc(
#         dodsrcFile = getDodsrcPath(), 
#         netrcFile  = getNetrcPath()
#         ):
#     if not os.path.exists(netrcFile):
#         return False
#     if not os.path.exists(dodsrcFile):
#         return False
    
#     with open(dodsrcFile, 'r') as f:
#         lines = f.readlines()
#         lines = [re.sub("http.*//", "", line) for line in lines]
    
#     return any(netrcFile in line for line in lines)

# def writeDodsrc(
#         netrcFile  = None,
#         dodsrcFile = ".dodsrc"
#         ):
    
#     if not netrcFile:
#         netrcFile = getNetrcPath()

#     # if not dodsrcFile:
#     #     dodsrcFile = getDodsrcPath()

#     # if (checkDodsrc(dodsrcFile, netrcFile) and not overwrite):
#     #     raise ValueError(f"{dodsrcFile} already exists. Set `overwrite=True` if you'd like to overwrite.")
    
#     dir = os.path.dirname(dodsrcFile)

#     # string = f'USE_CACHE=0\n\
#     #     MAX_CACHE_SIZE=20\n\
#     #     MAX_CACHED_OBJ=5\n\
#     #     IGNORE_EXPIRES=0\n\
#     #     CACHE_ROOT={dir}/.dods_cache/\n\
#     #     DEFAULT_EXPIRES=86400\n\
#     #     ALWAYS_VALIDATE=0\n\
#     #     DEFLATE=0\n\
#     #     VALIDATE_SSL=1\n\
#     #     HTTP.COOKIEJAR={dir}/.cookies\n\
#     #     HTTP.NETRC={netrcFile}'

#     string = f'USE_CACHE=0\n\
#         MAX_CACHE_SIZE=20\n\
#         MAX_CACHED_OBJ=5\n\
#         IGNORE_EXPIRES=0\n\
#         DEFAULT_EXPIRES=86400\n\
#         ALWAYS_VALIDATE=0\n\
#         DEFLATE=0\n\
#         VALIDATE_SSL=1\n\
#         HTTP.COOKIEJAR={dir}/.urs_cookies\n\
#         HTTP.NETRC={netrcFile}'
    
#     # create a dodsrc file
#     with open(os.path.expanduser(dodsrcFile), 'w') as f:
#         f.write(string)
        
#     # set the owner-only permission
#     os.chmod(dodsrcFile, 0o600)


#     return dodsrcFile

# def check_rc_files(
#         dodsrcFile = getDodsrcPath(), 
#         netrcFile  = getNetrcPath()
#         ):
    
#     if not checkDodsrc(dodsrcFile, netrcFile):
#         if checkNetrc(netrcFile):
#             print("Found Netrc file. Writing dodsrs file to: ", getDodsrcPath())
#             writeDodsrc(netrcFile, dodsrcFile)
#         else:
#             raise Exception("Netrc file not found. Please run writeNetrc() with earth data credentials.")