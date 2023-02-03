from objects.Operator import Operator
from functions.WebServerStuff.Web_Server import Web_Server

if __name__ == '__main__':
   i = int(input("Run Local or Web server:\n1 - Local\n2 - Web\n"))
   if i == 1:
      operator = Operator()
      operator.Start()
   elif i == 2:
      Web_Server()