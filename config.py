#!/usr/bin/env python3

import sys, argparse, configparser

'''
    Configuration file script that sets the following parameters:

    --verbose: if True, print verbose train/attack output,
    --train_fname: filename that contains training stats,
    --attack_fname: filename that contains attack stats,
    and any other added parameters.

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
    parser = argparse.ArgumentParser(description='Set configuration parameters of config.ini via CL.')
    parser.add_argument('-s', '--section', type=str,
                         help='section of config.ini in which to add parameter')
    parser.add_argument('--verbose', type=str, choices=['yes','no'], help='set verbose')
    parser.add_argument('-k', '--key', nargs='+', type=str, help='define key of config.ini')
    parser.add_argument('-v', '--val', nargs='+', type=str, help='define val of key')
    args = parser.parse_args()

    kwargs = dict()
    if args.verbose:
        kwargs['verbose'] = args.verbose
    if args.key and args.val:
        for key, val in zip(args.key, args.val):
            kwargs[key] = val
    write_config(args.section, **kwargs)

if __name__ == "__main__":
    # execute only if run as a script
    main()
