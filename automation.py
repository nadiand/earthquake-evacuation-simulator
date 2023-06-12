import evacuate

def main():
    stats = ""
    for i in range(10):
        stats += evacuate.main(['--input', 'in/merc.txt', '--numpeople', '100', '--max_time', '250', '--scaredness_rate', '0.5', '--follower_rate', '0.2'])
    
    print(stats)

if __name__ == '__main__':
    main()
    



