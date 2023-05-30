import evacuate

def main():
    stats = ""
    for i in range(2):
        stats += evacuate.main(['--input', 'in/merc.txt', '--numpeople', '60'])
    
    print(stats)

if __name__ == '__main__':
    main()
    



