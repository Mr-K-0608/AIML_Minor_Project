# Simple Grade Predictor based on Attendance and Assignment scores
Attendance = int(input("Enter Attendance percentage: "))
Assignment_score = int(input("Enter Assignment score: "))

final_score = (Attendance * 0.4) + (Assignment_score * 0.6)

# o.4 and 0.6 are the weights for attendance and assignment respectively

if final_score >= 85:
    print("Grade: A")
elif final_score >= 75:
    print("Grade: B")
elif final_score >= 65:
    print("Grade: C")
elif final_score >= 40:
    print("Grade: D")
else:
    print("Grade: F")

print(f"Final Score: {final_score}")
