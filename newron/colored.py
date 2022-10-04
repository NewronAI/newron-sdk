from newron.NColors import NColors


def success_print(text):
    print(NColors["SUCCESS"] + text + NColors["ENDC"])


def error_print(text):
    print(NColors["ERROR"] + text + NColors["ENDC"])


def warning_print(text):
    print(NColors["WARNING"] + text + NColors["ENDC"])


def header_print(text):
    print(NColors["HEADER"] + text + NColors["ENDC"])


def blue_print(text):
    print(NColors["BLUE"] + text + NColors["ENDC"])


def cyan_print(text):
    print(NColors["CYAN"] + text + NColors["ENDC"])


def bold_print(text):
    print(NColors["BOLD"] + text + NColors["ENDC"])


def underline_print(text):
    print(NColors["UNDERLINE"] + text + NColors["ENDC"])
