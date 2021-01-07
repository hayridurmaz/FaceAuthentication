import uuid

splitChar = ","


class User:
    def __init__(self, id, username, name):
        self.username = username
        self.name = name
        if id is None:
            self.id = uuid.uuid1()


def getUsers():
    userList = []
    file = open("text_data/users.csv", "r")
    userStrList = file.readlines()
    file.close()
    for str in userStrList:
        splited = str.split(splitChar)
        userList.append(User(splited[0], splited[1], splited[2]))
    return userList


def addUser(User):
    file = open("text_data/users.csv", "a")
    user_str = '{}{}{}{}{}\n'.format(User.id, splitChar, User.username, splitChar, User.name)
    file.write(user_str)
