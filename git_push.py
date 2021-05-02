import os

cmd = 'git add .'
temp = os.popen(cmd).read()
print(temp)
cmd = 'git status'
temp = os.popen(cmd).read()
print(temp)
inp = input("Message? Hit enter for simple 'bug fixes'\n\t")
if inp:
    message = inp
else:
    message = 'bug fixes'
cmd = 'git commit -m "{}"'.format(message)
temp = os.popen(cmd).read()
print(temp)
cmd = 'git push origin master'
temp = os.popen(cmd).read()
print(temp)