import chardet

def char_encode(location):

  # look at the first ten thousand bytes to guess the character encoding
  with open(location, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

  # check what the character encoding might be
  print(result)
  return result['encoding']
