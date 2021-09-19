#!/usr/bin/env python3

import sys, argparse, configparser

'''
    Configuration file script that sets the following parameters:

    --verbose: if True, print verbose train/attack output,
    --train_fname: filename that contains training stats,
    --attack_fname: filename that contains attack stats,
    and any other added parameters.

    This is an unofficial configuration file,
    please use another format for large-scale usage
    e.g. config.ini, config.yaml

    Usage via the command line.
'''
config = configparser.ConfigParser()
config.read('config.ini')
verbose = config.getboolean('general','verbose')
train_fname = config.get('general','train_fname')
attack_fname = config.get('general','attack_fname')

def write_config(section, **kwargs):
    '''
        Change configuration parameters.
    '''
    config = configparser.ConfigParser()
    config.read('config.ini')
    if not config.has_section(section):
        config.add_section(section)

    for key, val in kwargs.items():
        config[section][key] = val

    with open('config.ini', 'w') as configfile:
        config.write(configfile)

def main():
    parser = argparse.ArgumentParser(description='Set configuration parameters for main_script.py via CL.')
    parser.add_argument('-s', '--section', type=str,
                         help='section of config.ini in which to add parameter')
    parser.add_argument('-v', '--verbose', type=str, choices=['yes','no'], help='set verbose')
    parser.add_argument('--train', type=str, help='set train_fname')
    parser.add_argument('--attack', type=str, help='set attack_fname')
    parser.add_argument('--other', type=str, help='set other parameter')
    args = parser.parse_args()

    kwargs = dict()
    if args.verbose:
        kwargs['verbose']=args.verbose
    if args.train:
        kwargs['train']=args.train
    if args.attack:
        kwargs['attack']=args.attack
    if args.other:
        kwargs['other']=args.other

    write_config(args.section, **kwargs)

if __name__ == "__main__":
    # execute only if run as a script
    main()
