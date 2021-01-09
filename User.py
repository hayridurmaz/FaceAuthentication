from numpy import size

import config

splitChar = ","


class User:
    file = open(config.user_file, "r")
    userStrList = file.readlines()
    ID_seq = size(userStrList)

    def __init__(self, user_id, username, name):
        self.username = username
        self.name = name
        if user_id is None:
            self.id = User.ID_seq
            User.ID_seq = User.ID_seq + 1
        else:
            self.id = user_id


def getAllUsers():
    userList = []
    file = open(config.user_file, "r")
    userStrList = file.readlines()
    file.close()
    for string in userStrList:
        splited = string.split(splitChar)
        userList.append(User(splited[0], splited[1], splited[2]))
    return userList


def getUserByUsername(username):
    userList = getAllUsers()
    for user in userList:
        if user.username == username:
            return user
    return None


def getUserById(u_id):
    userList = getAllUsers()
    for user in userList:
        if user.id == str(u_id):
            return user
    return None


def addUser(user):
    file = open(config.user_file, "a")
    user_str = '{}{}{}{}{}\n'.format(user.id, splitChar, user.username, splitChar, user.name)
    file.write(user_str)
