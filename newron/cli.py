from fileinput import filename
from pprint import pprint
import click
import newron.utils
import os
import subprocess

@click.group()
def cli():
    pass

@cli.command()
def init():
    print("Welcome to Newron Project Initialisation")
    print("================================")

    print("Provide a name for this project")
    projectName= input()
    print("================================")

    print("Provide Location for Project | Both Relative and Absolute Paths are accepted.")
    print("This will initialise a project folder at this location")
    projectPath = input()
    os.chdir(projectPath)
    os.mkdir(projectName)
    os.chdir(f"./{projectName}")
    print("================================")

    print("Select Setup Format")
    formats = ["Jupyter Notebook", "Newron Project"]
    for index, format in enumerate(formats):
        print("["+str(index)+"]", format)
    format_response = input()
    format_response = newron.utils.inputParser(format_response, formats)
    print("You have selected", format_response)
    print("================================")
    filename = open("filename.py")

if __name__ == "__main__":
    cli()