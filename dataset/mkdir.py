import os
import json
output=open("list","w")
with open("ID_LIST.json") as json_file:
    json_data = json.load(json_file)
    for item in json_data:
        try:
            os.mkdir("{}".format(item["sno"]))
            os.mkdir("{0}/{1}".format(item["sno"],item["sname"]))
            output.write("{0}_{1}\n".format(item["sno"],item["sname"]))
        except:
            print(item)
            pass
output.close()
