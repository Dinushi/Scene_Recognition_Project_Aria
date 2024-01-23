import os
import json
import argparse
import sys

class ConfigParser:
    def __init__(self, config_file_path):
        self.configsMap = self.readInputConfigFile(config_file_path)

    # read a config json file at given path
    # return as a map config_key -> value
    def readInputConfigFile(self, config_file_path):
        with open(config_file_path, "r") as config_file:
            config = json.load(config_file)
        return config

    # return the requested config value from configMap
    # set a default if it does not available in config file
    def getConfigParam(self, param_name):
        configsMap = self.configsMap
        if param_name in configsMap:
            param_value = configsMap[param_name]
            #print("{} : {} ".format(param_name, param_value))
        else:
            if (param_name == "attack_list"):    
                param_value = []
            elif (param_name == "scans_root_path"):    
                param_value = ""

            else:
                param_value = None
        return param_value

def readProgrammeArgsForConfigFile(arg_parser):
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    in_path = parsed_args.input
    if not os.path.isabs(in_path):
        working_dir = os.getcwd()
        absolute_path = os.path.join(working_dir, in_path)
    else:
        absolute_path = in_path
    if os.path.exists(absolute_path):
        print("Input config json file exist at {}".format(in_path))
    else:
        print("No config file exist at given path!. Exist the program.")
        exit
    return absolute_path

# public
def generateAndAccessArgsForConfigFile():

    arg_parser = argparse.ArgumentParser(description='Description of the programme.')
    arg_parser.add_argument('input',
                    help='Path to the config json file.')
    input_config_file_path = readProgrammeArgsForConfigFile(arg_parser)
    return input_config_file_path