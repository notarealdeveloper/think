#!/usr/bin/env python3

__all__ = [
    'Year',
    'Month',
    'Day',
    'Weekday',
    'Date',
]

import datetime

from think import Digit, Digits, Str

class Year(Digits):

    class Millenium(Digit):
        ...
    class Century(Digit):
        ...
    class Decade(Digit):
        ...
    class Year(Digit):
        ...

    def __init__(self, year):
        if not year.isnumeric():
            raise TypeError(f"Not a year: {year!r}")
        if len(year) == 4:
            pass
        elif len(year) == 2:
            year = f"20{year}" if year[0] in '012' else f"19{year}"
        else:
            raise TypeError(year)
        m,c,d,y = year
        self.set(self.Millenium, m)
        self.set(self.Century, c)
        self.set(self.Decade, d)
        self.set(self.Year, y)


class Month(Digits):
    pass

class Day(Digits):
    pass

class Weekday(Str):
    INT_TO_NAME =  {1:'Monday', 2:'Tuesday', 3:'Wednesday',
                    4:'Thursday', 5:'Friday', 6:'Saturday',
                    7:'Sunday'}
    NAME_TO_INT = {v:k for k,v in INT_TO_NAME.items()}

    @classmethod
    def __object__(cls, arg):
        if isinstance(arg, int) and 1 <= arg <= 7:
            return cls.INT_TO_NAME[arg]
        elif isinstance(arg, str) and arg in '1234567':
            return cls.INT_TO_NAME[int(arg)]
        elif isinstance(arg, str) and arg in cls.NAME_TO_INT:
            return arg
        else:
            raise ValueError(arg)


class Date(Str):

    def __init__(self, date, set_year=True, set_month=True, set_day=True, set_weekday=True):
        data = self.parse(date)
        if (year := data.get('year')) and set_year:
            self.set(Year, year)
        if (month := data.get('month')) and set_month:
            self.set(Month, month)
        if (day := data.get('day')) and set_day:
            self.set(Day, day)
        if (weekday := data.get('weekday')) and set_weekday:
            weekday = weekday() if callable(weekday) else weekday
            self.set(Weekday, weekday)

    @classmethod
    def date_object_to_dict(cls, date):
        d = {k: getattr(date, k) for k in ('year', 'month', 'day', 'weekday')}
        if callable(d['weekday']): # for 2/3 methods, it's a function
            d['weekday'] = d['weekday']()+1
        d = {k: str(v) for k,v in d.items()}
        lengths = {'year': 4, 'month': 2, 'day': 2, 'weekday': 1}
        for k in d:
            d[k] = d[k].zfill(lengths[k])
        return d

    @classmethod
    def suspicious_year(cls, input, output):
        this_year = datetime.date.today().year
        return  output['year'] == this_year \
        and     this_year not in re.findall(r'\d{4}', input)

    @classmethod
    def parse_with_timestring(cls, input):
        import tmstr
        date = tmstr.Date(input)
        output = cls.date_object_to_dict(date)
        if cls.suspicious_year(input, output):
            # gives the present year when no year is found
            output['year'] = None
            output['weekday'] = None
        return output

    @classmethod
    def parse_with_dateparser(cls, input):
        import dateparser
        date = dateparser.parse(input)
        output = cls.date_object_to_dict(date)
        if cls.suspicious_year(input, output):
            # dateparser gives the present year when no year is found
            output['year'] = None
            output['weekday'] = None
        return output

    @classmethod
    def parse_with_datetime_module(cls, input):
        date = datetime.date.fromisoformat(input)
        output = cls.date_object_to_dict(date)
        return output

    def parse(cls, input):
        try:
            return cls.parse_with_timestring(input)
        except:
            pass
        try:
            return cls.parse_with_dateparser(input)
        except:
            pass
        try:
            return cls.parse_with_datetime_module(input)
        except:
            pass
        raise ValueError(input)

