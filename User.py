import uuid

from numpy import size

import config


class User:
    file = open(config.user_file, "r")
    userStrList = file.readlines()
    ID_seq = size(userStrList)

    def __init__(self, id, username, name):
        self.username = username
        self.name = name
        if id is None:
            self.id = User.ID_seq
            User.ID_seq = User.ID_seq + 1
        else:
            self.id = id


def getAllUsers():
    userList = []
    file = open(config.user_file, "r")
    userStrList = file.readlines()
    file.close()
    for str in userStrList:
        splited = str.split(splitChar)
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


def addUser(User):
    file = open(config.user_file, "a")
    user_str = '{}{}{}{}{}\n'.format(User.id, splitChar, User.username, splitChar, User.name)
    file.write(user_str)


splitChar = ","
