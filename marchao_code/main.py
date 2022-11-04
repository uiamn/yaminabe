table = {
    'a': '四条烏丸',
    'b': '梅田',
    'g': '難波',
    'd': '吉祥寺',
    'e': '神戸',
    'z': '新宿',
    'h': '難波',
    'q': '池袋',
    'i': '秋葉原',
    'k': '大和八木',
    'l': '北千住',
    'm': '横浜',
    'n': '川口',
    'x': '天王寺',
    'o': '梅田',
    'p': '船橋',
    'r': '名古屋駅前',
    's': '千葉中央',
    't': '川崎',
    'u': '赤羽',
    'f': '広島',
    'c': '名古屋栄',
    'y': '仙台',
    'w': '堺東',
    'v': '新越谷',
    'j': '札幌'
}

trans = str.maketrans(table)

print(input().lower().translate(trans))