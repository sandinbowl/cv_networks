"""
给定字符串s，求s中长度最长的回文，并return最长回文（之一）。
"""
def find_max_plalindrome(s):
    possible_ans = []
    for i in range(len(s)):
        for j in range(len(s)):
            if j <= i:
                continue
            elif j == (i + 1):
                possible_ans.append(s[i])
            else:
                poss_plalindrome = s[i:j]
                temp = [ele for ele in poss_plalindrome]
                temp_copy = temp[:]
                temp_copy.reverse()
                if temp == temp_copy:
                    possible_ans.append(poss_plalindrome)
    largest_len = -1
    result = ''
    for word in possible_ans:
        if largest_len < len(word):
            largest_len = len(word)
            result = word
    return result


if __name__ == '__main__':
    s = '12345654121'
    ans = find_max_plalindrome(s)
    print(ans)
