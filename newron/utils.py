"""
CLI Utils
"""

def inputParser(userInput, options):
    try:
        index = int(userInput)
        return(options[index])
    except:
        for option in options:
            if option.lower() == userInput.lower():
                return(option)
                break



def writepy(fileName):
    contents = []
    print("Welcome to .py file setup")
    print("================================")

    print("Please Select a Framework from the List Provided Below")
    frameworks = ["sklearn", "keras"]
    for index, framework in enumerate(frameworks):
        print("["+str(index)+"]", framework)
    framework_response = input()
    framework_response = inputParser(framework_response, formats)
    print("You have selected", framework_response)
    print("================================")
    
