"""Kant van een plaatscan (0.5.56), feit: de nummers op een label zijn de nummers van die kant volgens Discogs.

Who's Your Momma 2016 zilver: wit label op transparant vinyl, "SIDE TWO" onderstreept. Op de verkleinde kopie las de
tekstherkenning "SIDP. ONF", uit de scan zelf "SIDE TW" (de streep valt over de O). De kant werd niet gelezen en scan
0014 (label kant B) kwam op scanvolgorde bij kant A. Nu: (1) een kantwoord waarvan alleen de laatste letter(s)
ontbreken telt als het begin eenduidig is; (2) staan minstens twee titels van één kant op het label en geen van een
andere kant, dan is het die kant.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Resources', 'Tools'))
import scan_vakken as V
V.SIDE_TRACKS.clear()
V.SIDE_TRACKS.update({'sidea': [V.norm_title(t) for t in ["If I Go", "Might As Well", "Make It Rain", "Modern World", "I Don't Wanna Hurt"]],
                      'sideb': [V.norm_title(t) for t in ["Good God", "The Difference", "Whatever You Say", "Ball And Chain", "Daze", "If You Were Mine"]]})
# gelezen regels van het label van kant B (zilver 0014, met leesfouten zoals de tekstherkenning ze gaf)
b = ['ANOUK', "WHO'S YOUR MOMMA", "I GOOD GOD 2.'36", '2 THE DIFFERENCE 3:05', '3 WHATEVER YOU SAY 3..15', '5 DAZE 3..36']
a = ['SIDE ONF', 'IIFIGO424', '2 MIGHT AS WELL 3:09', '4 MODERN WORLD']
assert V.tracks_side(b) == 'sideb', V.tracks_side(b)
assert V.tracks_side(a) == 'sidea', V.tracks_side(a)
assert V.tracks_side(['ANOUK', '1 GOOD GOD']) is None                    # één titel is niet genoeg
assert V.tracks_side(b + ['4 MODERN WORLD']) is None                     # titels van twee kanten: geen kant
V.SIDE_TRACKS.clear()
assert V.tracks_side(b) is None                                          # zonder Discogs-tracklist geen kant
print('kant van een plaatscan: tracklist van het label wijst de kant aan (Discogs), ook als SIDE niet te lezen is')
