import pickle
from functions.singleLossFunctions.Single_Loss_Function_Choice import Single_Loss_Function_Choice
def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except pickle.PicklingError:
        return False
    return True

# Example usage:
print(is_picklable(Single_Loss_Function_Choice))  # Replace with your function or object
#print(is_picklable(SomeData))  # Replace with your data
