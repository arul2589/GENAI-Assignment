class GradeSystem:

    def calculate_grade (self , mark):
        if mark >= 90:
            return  "A"

        elif mark >= 80:
            return  "B"

        elif mark >= 70:
            return  "C"

        elif mark >= 60:
            return  "D"

        else:
            return  "E"

    def run(self):
        while True:
            try:
                mark = int(input("Enter your mark (0 - 100): "))

                if mark < 0 or mark > 100:
                    print ("Enter mark between 0 and 100.")
                    continue

                grade = self.calculate_grade(mark)
                print(f"Mark: {mark} -> Grade: {grade}")
                break

            except ValueError:
                print("Enter a valid whole-number mark.")


grader = GradeSystem()
grader.run()