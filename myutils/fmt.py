import time


HOUR = 3600
DAY = HOUR * 24
MONTH = DAY * 31
YEAR = DAY * 365

def fmt_timespan(timespan):
    if timespan <= 600:
        return '%.4gs' % timespan
    else:
        if timespan < HOUR:
            return time.strftime('%Mm-%Ss', time.gmtime(timespan))
        elif timespan < DAY:
            return time.strftime('%Hh-%Mm-%Ss', time.gmtime(timespan))
        elif timespan < MONTH:
            return time.strftime('%dd-%Hh-%Mm-%Ss', time.gmtime(timespan - DAY))
        elif timespan < YEAR:
            return time.strftime('%mm-%dd-%Hh-%Mm-%Ss', time.gmtime(timespan - MONTH))
        else:
            return '%dy-' % (timespan // 31536000) + time.strftime('%mm-%dd-%Hh-%Mm-%Ss', time.gmtime(timespan - YEAR))

def fmt_floats(floats, fmt='%.2f'):
    return list(map(lambda x: fmt % x, floats))

def float_no_zeros(v, n):
    fmt = f'%.{n}f'
    res = (fmt % v).rstrip('0').rstrip('.')
    if res == '-0': res = '0'
    return res

def now_datetime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


if __name__ == '__main__':

    print(fmt_timespan(DAY + 1000))
    print(fmt_timespan(MONTH + 1000))
    print(fmt_timespan(10800))
