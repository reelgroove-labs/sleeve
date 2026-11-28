"""Sleeve 0.5.60: the name pattern from Settings (SLEEVE_NAME_TEMPLATE) decides the order of the parts; the default stays the collection scheme."""
import os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Resources/Tools'))
import scan_naam as n
release={'year':2016,'released':'2016-06-20','title':'Hotel New York','artists':[{'name':'Anouk'}],
         'formats':[{'name':'Vinyl','descriptions':['LP','Album','Red']}],'labels':[{'name':'Music On Vinyl','catno':'MOVLP1575'}]}
n.get=lambda url:release
url='https://www.discogs.com/release/1'
os.environ.pop('SLEEVE_NAME_TEMPLATE',None)
assert n.naam(url,'sidea')=='2016_hotel_new_york_vinyl_musiconvinyl_red_sidea', n.naam(url,'sidea')
os.environ['SLEEVE_NAME_TEMPLATE']='title_catno_colour'
assert n.naam(url,'sidea')=='hotel_new_york_movlp1575_red_sidea', n.naam(url,'sidea')
os.environ['SLEEVE_NAME_TEMPLATE']='artist_title_format_date_part'
assert n.naam(url,'front')=='anouk_hotel_new_york_vinyl_20160620_front', n.naam(url,'front')
os.environ['SLEEVE_NAME_TEMPLATE']='nonsense'
assert n.naam(url,'sidea')=='2016_hotel_new_york_vinyl_musiconvinyl_red_sidea'
print('Name pattern from Settings works; the default name stays the same.')
