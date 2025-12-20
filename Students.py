Students={
    "kevin": [12, 96],
    "Don" : [12, 75],
    "Alice" : [12, 88]

}
counter = 0
total = 0 
for student in Students :
    total += int(Students[student][1])
    counter +=1
print("Average score:", total / counter)
