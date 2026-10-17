while True:
     

    try:
        mark = int(input("Enter your mark (0 - 100): "))

        if mark < 0 or mark >100:
            raise ValueError

        if mark >=90:
             grade = "A"
            
        elif mark >=80:
            grade = "B"

        elif mark >=70:
            grade = "C"

        elif mark >=60:
            grade = "D"

        else:
            grade = "E"

        print (f"Mark: {mark} -> Grade: {grade}")

        break

    except ValueError:
        print ("Enter a valid whole-number mark between 0 and 100")

    
