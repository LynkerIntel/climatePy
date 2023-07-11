import os
import platform
import re

def whatOS():
    os_name = platform.system().lower()
    if os_name == 'darwin':
        os_name = 'osx'
    return os_name

def getNetrcPath():
    home = os.environ['HOME']
    if whatOS() == "windows":
        return os.path.join(home, "_netrc")
    else:
        return os.path.join(home, ".netrc")

def getDodsrcPath():
    home = os.getenv("HOME")
    if whatOS() == "windows":
        return os.path.join(home, '_dodsrc')
    else:
        return os.path.join(home, '.dodsrc')

def writeNetrc(
        login     = None, 
        password  = None, 
        machine   = 'urs.earthdata.nasa.gov',
        netrcFile = None, 
        overwrite = False
        ):
    
    """Write netrc file"""

    if not netrcFile:
        netrcFile = getNetrcPath()
    
    if login is None or password is None:
        raise ValueError("Login/Password is missing. If you don't have an account please register at:\nhttps://urs.earthdata.nasa.gov/users/new")
    
    if os.path.exists(netrcFile) and not overwrite:
        raise FileExistsError("'" + netrcFile + "' already exists. Set `overwrite=True` if you'd like to overwrite.")
    
    string = f"\nmachine {machine}\nlogin {login}\npassword {password}"
    
    with open(os.path.expanduser(netrcFile), 'a') as f:
        f.write(string)
    
    os.chmod(netrcFile, 0o600)
    
    return netrcFile

def checkNetrc(
        netrcFile = getNetrcPath(),
        machine   = "urs.earthdata.nasa.gov"
        ):
    """Check netrc file"""

    if not os.path.exists(netrcFile):
        return False
    
    with open(netrcFile) as f:
        lines = f.readlines()

    lines = [line.replace("http.*//", "") for line in lines]

    return any(machine in line for line in lines)

def checkDodsrc(
        dodsrcFile = getDodsrcPath(), 
        netrcFile  = getNetrcPath()
        ):
    if not os.path.exists(netrcFile):
        return False
    if not os.path.exists(dodsrcFile):
        return False
    
    with open(dodsrcFile, 'r') as f:
        lines = f.readlines()
        lines = [re.sub("http.*//", "", line) for line in lines]
    
    return any(netrcFile in line for line in lines)

def writeDodsrc(
        netrcFile  = None,
        dodsrcFile = ".dodsrc"
        ):
    
    if not netrcFile:
        netrcFile = getNetrcPath()

    # if not dodsrcFile:
    #     dodsrcFile = getDodsrcPath()

    # if (checkDodsrc(dodsrcFile, netrcFile) and not overwrite):
    #     raise ValueError(f"{dodsrcFile} already exists. Set `overwrite=True` if you'd like to overwrite.")
    
    dir = os.path.dirname(dodsrcFile)

    # string = f'USE_CACHE=0\n\
    #     MAX_CACHE_SIZE=20\n\
    #     MAX_CACHED_OBJ=5\n\
    #     IGNORE_EXPIRES=0\n\
    #     CACHE_ROOT={dir}/.dods_cache/\n\
    #     DEFAULT_EXPIRES=86400\n\
    #     ALWAYS_VALIDATE=0\n\
    #     DEFLATE=0\n\
    #     VALIDATE_SSL=1\n\
    #     HTTP.COOKIEJAR={dir}/.cookies\n\
    #     HTTP.NETRC={netrcFile}'

    string = f'USE_CACHE=0\n\
        MAX_CACHE_SIZE=20\n\
        MAX_CACHED_OBJ=5\n\
        IGNORE_EXPIRES=0\n\
        DEFAULT_EXPIRES=86400\n\
        ALWAYS_VALIDATE=0\n\
        DEFLATE=0\n\
        VALIDATE_SSL=1\n\
        HTTP.COOKIEJAR={dir}/.urs_cookies\n\
        HTTP.NETRC={netrcFile}'
    
    # create a dodsrc file
    with open(os.path.expanduser(dodsrcFile), 'w') as f:
        f.write(string)
        
    # set the owner-only permission
    os.chmod(dodsrcFile, 0o600)


    return dodsrcFile

def check_rc_files(
        dodsrcFile = getDodsrcPath(), 
        netrcFile  = getNetrcPath()
        ):
    
    if not checkDodsrc(dodsrcFile, netrcFile):
        if checkNetrc(netrcFile):
            print("Found Netrc file. Writing dodsrs file to: ", getDodsrcPath())
            writeDodsrc(netrcFile, dodsrcFile)
        else:
            raise Exception("Netrc file not found. Please run writeNetrc() with earth data credentials.")