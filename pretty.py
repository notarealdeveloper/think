#!/usr/bin/env python3

""" The pretty module """

__all__ = [
    'colors',
]

# colors

class color(str):
    end = "\033[00m"
    def __init__(self, code):
        self.code = code
    def __call__(self, string):
        return f"{self.code}{string}{self.end}"

class colors:
    black   = color("\033[01;30m")
    red     = color("\033[01;31m")
    green   = color("\033[01;32m")
    yellow  = color("\033[01;33m")
    blue    = color("\033[01;34m")
    purple  = color("\033[01;35m")
    cyan    = color("\033[01;36m")
    gray    = color("\033[01;37m")
    white   = color("\033[01;39m")

