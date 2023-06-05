import evacuate

def main():
    stats = ""
    for i in range(5):
        stats += evacuate.main(['--input', 'in/merc.txt', '--numpeople', '60', '--max_time', '250'])
    
    print(stats)

if __name__ == '__main__':
    main()
    



