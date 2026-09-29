# grade_system.py

while True:
    try:
        mark = float(input("Enter a mark (0-100): "))

        # Check whether the mark is within the valid range
        if mark < 0 or mark > 100:
            print("Invalid mark. Please enter a number between 0 and 100.")
            continue

        # Determine the grade
        if mark >= 90:
            grade = "A"
        elif mark >= 80:
            grade = "B"
        elif mark >= 70:
            grade = "C"
        elif mark >= 60:
            grade = "D"
        else:
            grade = "E"

        print(f"Mark: {mark}")
        print(f"Grade: {grade}")
        break

    except ValueError:
        print("Invalid input. Please enter a number.")