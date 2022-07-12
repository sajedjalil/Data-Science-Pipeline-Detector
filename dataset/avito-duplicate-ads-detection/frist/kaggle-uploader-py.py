"""
Script to automatic submission uploading to Kaggle

Very simple tool to make an automatic upload generated submissions to kaggle.

Sample Configuration File:
[DEFAULT]
username=johndoe@example.com
password=mYpaSsworD

Example usage with config:
$ python kaggle-uploader.py -c ~/.kaggle.cfg \
    --url https://www.kaggle.com/c/avito-duplicate-ads-detection/submissions/attach \
    --description 'a couple of words about submission' somefile.csv

Example usage without config file:
$ python kaggle-uploader.py -u johndoe@example.com -p mYpaSsworD \
    --url https://www.kaggle.com/c/avito-duplicate-ads-detection/submissions/attach \
    --description 'a couple of words about submission' somefile.csv
""" 

import ConfigParser
import argparse
import cookielib
import os

from mechanize import Browser

parser = argparse.ArgumentParser(description='Upload file to kaggle.')
parser.add_argument('file',
                    help='file to upload')
parser.add_argument('-c', '--config', dest='config',
                    help='configuration file')
parser.add_argument('-u', '--username', dest='username',
                    help='username to login')
parser.add_argument('-p', '--password', dest='password',
                    help='password to login')
parser.add_argument('--url', dest='url', required=True,
                    help='competition\'s upload url')
parser.add_argument('--description', dest='description',   
                    help='submission\'s description')

args = parser.parse_args()

if args.config is not None:
    config = ConfigParser.ConfigParser()
    config.read(args.config)

    try:
        username = config.get('DEFAULT', 'username')
    except ConfigParser.NoOptionError:
        username = None

    try:
        password = config.get('DEFAULT', 'password')
    except ConfigParser.NoOptionError:
        password = None

username = args.username if args.username is not None else username
password = args.password if args.password is not None else password

if any((
        args.file is None,
        username is None,
        password is None,
        args.url is None
    )):
    parser.print_help()
    exit()

br = Browser()
cj = cookielib.LWPCookieJar()
br.set_cookiejar(cj)

br.open('https://www.kaggle.com/account/login')
br.select_form(nr=0)
br['UserName'] = username
br['Password'] = password
br.submit(nr=0)

br.open(args.url)
br.select_form(nr=0)

br.add_file(open(args.file), 'application/octet-stream', os.path.basename(args.file), name='SubmissionUpload')
if args.description is not None:
    br['SubmissionDescription'] = args.description
br.submit(nr=0)

