import re
from sys import exit


with open("partdiff-real.data", "r") as file:
    lines = [line.rstrip() for line in file.readlines()]
    
    header = lines[0].split(" ")
    rows = []
    new_data = []

    for line in lines[1:]:
        values = [float(value) for value in line.split(" ")]
        rows.append(values)
        new_data.append([])

    new_header = []
    normal_value_index = -1

    for index in range(len(header)):
        head = header[index]

        if head == "threads":
            new_header.append(head)
            for row_index in range(len(rows)):
                row = rows[row_index]
                datum = row[index]
                new_row = new_data[row_index]
                new_row.append(int(datum))
            continue
        
        if head == "pd-error" or head == "pd" or head == "pd-p-error":
            continue

        if head == "pd-p":
            normal_value_index = index
            continue

        if normal_value_index < 0:
            print("Normal Value not defined, Columns are not ordered correctly")
            exit(1)

        new_header.append(head)
        
        if "error" in head:
            for row_index in range(len(rows)):
                row = rows[row_index]
                normal_value = row[normal_value_index]
                datum = row[index]

                deviant_value = normal_value / (row[index-1] - datum)
                new_row = new_data[row_index]
                try:
                    new_error = abs(new_row[len(new_row) - 1] - deviant_value)
                    new_row.append(new_error) 
                except:
                    print(index, new_row)
        else:
            for row_index in range(len(rows)):
                row = rows[row_index]
                datum = row[index]
                normal_value = row[normal_value_index]

                new_datum = normal_value / datum
                new_row = new_data[row_index]
                new_row.append(new_datum)

    with open("partdiff-speedup-relative.data", "w") as out:
        out_lines = []
        out_lines.append(" ".join(new_header) + "\n")

        for row in new_data:
            string = ""
            for new_datum in row:
                string += " " + str(new_datum)
            out_lines.append(string.strip() + "\n")
        out.writelines(out_lines)
