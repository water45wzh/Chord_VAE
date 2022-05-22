from music21 import *

sc = converter.parse('chord_set/endre---kallocain_chorus.mid')

for i in sc[0][4:]:
    print(i.normalOrder)

#sc.show()