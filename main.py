# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib
import pickle
matplotlib.use('TkAgg')

outliers = [
    [52144, 627, 446],
    [54502, 2658, 164],
    [54568, 2524, 115],
    [53794, 2217, 589],
    [52522, 1057, 58],
    [54504, 2660, 167],
    [51821, 359, 2],
    [51876, 437, 579],
    [51665, 311, 280],
    [52738, 1291, 349],
    [53558, 2207, 454],
    [51993, 334, 501],
    [53147, 1611, 219],
    [52381, 886, 221],
    [53714, 2106, 265],
    [51959, 541, 214],
    [51999, 362, 26],
    [54453, 1875, 483],
    [52734, 1289, 9],
    [52250, 412, 265],
    [51990, 310, 97],
    [51821, 413, 355],
    [53090, 1370, 72],
    [51883, 436, 490],
    [53357, 1749, 342],
    [52762, 1325, 634],
    [54234, 2663, 56],
    [53799, 2230, 638],
    [53357, 2081, 637],
    [53149, 1679, 616],
    [52078, 371, 144],
    [52199, 659, 101],
    [54535, 2765, 124],
    [53503, 2024, 185],
    [51630, 266, 209],
    [53442, 1768, 169],
]


def sql_query():
    q = """
SELECT    p.objid, p.ra,p.dec, s.plate, s.mjd, s.fiberid, s.run2d
FROM PhotoObj AS p
   JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE  
 """
    first = True
    for o in outliers:
        if first:
            s = ""
            first = False
        else:
            s = " OR "
        s = s + 's.mjd = {} AND s.plate = {} AND s.fiberid = {}'.format(o[0], o[1], o[2])
        q = q + s
    q + ";"
    print(q)

def test():
    f = 'test_set_dec.csv'
    df = pandas.read_csv(f)
    return df

def training():
    f = 'training_set_dec.csv'
    df = pandas.read_csv(f)
    return df

def addDec():
    file="combined_galaxies_quasars_bins1000_wl3388-8318_minz005_maxz036_minSN50_new_relabeled.pkl"

    out = pickle.load(open(file,'rb'))

    # out = pickle.load(file)
    # train,valid,test,le = pickle.load(file,'rb')
    fs = [ 'training_set.csv', 'validation_set.csv','test_set.csv']
    for f, meta in zip(fs,out):
        df = pandas.read_csv(f)
        # ra=[]
        dec=[]
        for i,row in df.iterrows():
            w=numpy.logical_and.reduce(
                [meta["MJD"]==row['MJD'], meta["plate"]==row['plate'], meta['fiber']==row["fiber"]])
            w=numpy.where(w == True)
            # ra.append(meta['RA'][w[0]][0])
            dec.append(meta['DEC'][w[0]][0])

        df['DEC']=dec
        df.to_csv(f.replace('.csv','_dec.csv'))

def validation():
    f = 'validation_set_dec.csv'
    df = pandas.read_csv(f)
    return df

def graur():
    f = 'graur.txt'
    df = pandas.read_csv(f, sep=" ")
    return df

def supernova():
    gdf=graur()

    df=pandas.concat((test(),training(),validation()))

    matches=pandas.DataFrame()
    for index, row in gdf.iterrows():
        plate,mjd,fiber = row['SDSS_ID'].split('-')
        plate= int(plate)
        fiber = int(fiber)
        mjd = int(mjd)
        match = df[numpy.logical_and.reduce([df["MJD"]==mjd, df["plate"]==plate, df["fiber"]==fiber])]
        matches=matches.append(match)

    plt.hist([matches['logp'],df['logp']],label=['SN','All'],density=True,range=[-30,20])
    plt.xlabel('logp')
    plt.legend()
    plt.show()
    plt.hist([matches['recon_error'],df['recon_error']],bins=numpy.logspace(numpy.log10(0.5),numpy.log10(20.0), 10),label=['SN','All'],density=True)
    plt.xlabel('recon_error')
    plt.xscale('log')
    plt.legend()
    plt.show()

def quickView(selection='logp'):
    f = 'test_set_dec.csv'
    df = pandas.read_csv(f)
    ascending=dict()
    ascending['logp']=True
    ascending['recon_error']=False
    for i in range(2):
        print(i)
        _d = df.loc[df['new_label'] == i]
        _d = _d.sort_values(selection,ascending=ascending[selection])

        counter=0
        for index, row in _d.iterrows():
            print(row[selection])
            # print("http://skyserver.sdss.org/dr17/en/tools/quicklook/summary.aspx?plate={}&mjd={}&fiber={}".format(
            #     int(row['plate']), int(row['MJD']), int(row['fiber'])))
            # print("https://www.legacysurvey.org/viewer-desi?ra={}&dec={}&zoom=16".format(
            #     row['RA'], row['DEC']))

            counter=counter+1
            if counter ==6:
                break
            # print("https://dr17.sdss.org/optical/spectrum/view?plate={}&mjd={}&fiber={}".format(
            #     int(row['plate']), int(row['MJD']), int(row['fiber'])))

def compareList():
    f = 'test_set_no_QSO_relabeled_retrained.csv'
    df = pandas.read_csv(f)
    for i in range(3,4):
        print(i)
        _d = df.loc[df['subclass'] == i]
        _d = _d.sort_values('logp')
        counter=0
        for index, row in _d.iterrows():
            print(int(row['plate']), int(row['MJD']), int(row['fiber']), row['logp'])
            counter=counter+1
            if counter ==12:
                break
            # print("https://dr17.sdss.org/optical/spectrum/view?plate={}&mjd={}&fiber={}".format(
            #     int(row['plate']), int(row['MJD']), int(row['fiber'])))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # supernova()
    # addDec()
    quickView('recon_error')
    # compareList()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
