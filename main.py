# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib
import pickle

matplotlib.use('TkAgg')

# {6.0: 'GAL STARFORMING NOT BROADLINE',
#  0.0: 'GAL NO SUBCLASS NOT BROADLINE',
#  4.0: 'GAL STARBURST NOT BROADLINE',
#  1.0: 'GAL NO SUBCLASS BROADLINE',
#  13.0: 'QSO STARBURST BROADLINE',
#  2.0: 'GAL AGN NOT BROADLINE',
#  15.0: 'QSO STARFORMING BROADLINE',
#  9.0: 'QSO NO SUBCLASS BROADLINE',
#  7.0: 'GAL STARFORMING BROADLINE',
#  8.0: 'QSO NO SUBCLASS NOT BROADLINE',
#  11.0: 'QSO AGN BROADLINE',
#  3.0: 'GAL AGN BROADLINE',
#  10.0: 'QSO AGN NOT BROADLINE',
#  5.0: 'GAL STARBURST BROADLINE',
#  14.0: 'QSO STARFORMING NOT BROADLINE',
#  12.0: 'QSO STARBURST NOT BROADLINE'}

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
    file = "combined_galaxies_quasars_bins1000_wl3388-8318_minz005_maxz036_minSN50_new_relabeled.pkl"

    out = pickle.load(open(file, 'rb'))

    # out = pickle.load(file)
    # train,valid,test,le = pickle.load(file,'rb')
    fs = ['training_set.csv', 'validation_set.csv', 'test_set.csv']
    for f, meta in zip(fs, out):
        df = pandas.read_csv(f)
        # ra=[]
        dec = []
        for i, row in df.iterrows():
            w = numpy.logical_and.reduce(
                [meta["MJD"] == row['MJD'], meta["plate"] == row['plate'], meta['fiber'] == row["fiber"]])
            w = numpy.where(w == True)
            # ra.append(meta['RA'][w[0]][0])
            dec.append(meta['DEC'][w[0]][0])

        df['DEC'] = dec
        df.to_csv(f.replace('.csv', '_dec.csv'))


def validation():
    f = 'validation_set_dec.csv'
    df = pandas.read_csv(f)
    return df


def graur():
    f = 'graur.txt'
    df = pandas.read_csv(f, sep=" ")
    return df


def supernova():
    gdf = graur()

    df = pandas.concat((test(), training(), validation()))

    matches = pandas.DataFrame()
    for index, row in gdf.iterrows():
        plate, mjd, fiber = row['SDSS_ID'].split('-')
        plate = int(plate)
        fiber = int(fiber)
        mjd = int(mjd)
        match = df[numpy.logical_and.reduce([df["MJD"] == mjd, df["plate"] == plate, df["fiber"] == fiber])]
        matches = matches.append(match)

    plt.hist([matches['logp_marg_corr'], df['logp']], label=['SN', 'All'], density=True, range=[-30, 20])
    plt.xlabel('logp')
    plt.legend()
    plt.show()
    plt.hist([matches['recon_error'], df['recon_error']], bins=numpy.logspace(numpy.log10(0.5), numpy.log10(20.0), 10),
             label=['SN', 'All'], density=True)
    plt.xlabel('recon_error')
    plt.xscale('log')
    plt.legend()
    plt.show()


def tails():
    selections = ['logp_marg_corr', 'recon_error']
    # selections = ['recon_error']

    tags = ['test', 'training', 'validation']

    for s in selections:
        holder = []
        for t in tags:
            f = '{}_set_dec.csv'.format(t)
            df = pandas.read_csv(f)
            _d = df.loc[numpy.logical_or(df['new_label'] == 0, df['new_label'] == 1)]
            holder.append(_d[s].to_numpy())

        if s == 'logp_marg_corr':
            mi = numpy.min([h.min() for h in holder])
            b = numpy.linspace(mi, mi + 15, 5)
            b = numpy.linspace(-40, 25, 25)
            plt.hist(holder, bins=b, label=tags)
            plt.xlabel(s)

        elif s == 'recon_error':
            b = numpy.logspace(1e-5, 0.01, 5)
            b = numpy.linspace(0.3, 1, 20)
            # for i in range(len(holder)):
            #     holder[i]=holder[i]-1
            plt.hist(holder, bins=b, label=tags)
            # plt.xscale('log')
            plt.xlabel(s)

        plt.legend()
        plt.yscale('log')
        plt.xlabel(s)
        plt.show()


def outliers():
    selections = ['logp_marg_corr', 'recon_error']
    # selections = ['recon_error']

    tags = ['test', 'validation']

    bins = dict()
    bins['logp_marg_corr'] = numpy.linspace(-80, -10, 15)
    bins['recon_error'] = numpy.linspace(10, 40, 15)

    dfs = []
    for t in tags:
        f = '{}_set_dec.csv'.format(t)
        dfs.append(pandas.read_csv(f))
        # _d = df.loc[numpy.logical_or(df['new_label'] == 0, df['new_label'] == 1)]
        # holder.append(_d[s].to_numpy())
    dfs = pandas.concat(dfs)
    print(type(dfs.new_label))
    dfs['galaxy'] = dfs.new_label // 8 == 0
    dfs['broadline'] = ((dfs.new_label % 2) == 1)
    dfs['generic'] = ((dfs.new_label % 8) // 2 == 0)
    dfs['AGN'] = ((dfs.new_label % 8) // 2 == 1)
    dfs['starburst'] = ((dfs.new_label % 8) // 2 == 2)
    dfs['starforming'] = ((dfs.new_label % 8) // 2 == 3)

    # #absolultely worst
    # print("ABSOLUTELY WORST")
    # _d = dfs.sort_values('logp_marg_corr')
    # counter=0
    # for index, row in _d.iterrows():
    #     # print(row[selection])
    #     print(row['logp_marg_corr'],row['new_label'])
    #     print("http://skyserver.sdss.org/dr17/en/tools/quicklook/summary.aspx?plate={}&mjd={}&fiber={}".format(
    #         int(row['plate']), int(row['MJD']), int(row['fiber'])))
    #     print("https://www.legacysurvey.org/viewer-desi?ra={}&dec={}&zoom=16".format(
    #         row['RA'], row['DEC']))
    #     counter = counter + 1
    #     if counter==5:
    #         break
    #
    #
    # _d = dfs.sort_values('recon_error',ascending=False)
    # counter = 0
    # for index, row in _d.iterrows():
    #     # print(row[selection])
    #     print(row['recon_error'],row['new_label'])
    #     print("http://skyserver.sdss.org/dr17/en/tools/quicklook/summary.aspx?plate={}&mjd={}&fiber={}".format(
    #         int(row['plate']), int(row['MJD']), int(row['fiber'])))
    #     print("https://www.legacysurvey.org/viewer-desi?ra={}&dec={}&zoom=16".format(
    #         row['RA'], row['DEC']))
    #     counter = counter + 1
    #     if counter == 5:
    #         break

    # Worst Galaxy logp
    # print("WORST GALAXY LOGP")
    # _d = dfs.loc[dfs['galaxy']]
    # _d = _d.sort_values('logp_marg_corr')
    # counter = 0
    # for index, row in _d.iterrows():
    #     # print(row[selection])
    #     print(row['logp_marg_corr'],row['new_label'])
    #     print("http://skyserver.sdss.org/dr17/en/tools/quicklook/summary.aspx?plate={}&mjd={}&fiber={}".format(
    #         int(row['plate']), int(row['MJD']), int(row['fiber'])))
    #     print("https://www.legacysurvey.org/viewer-desi?ra={}&dec={}&zoom=16".format(
    #         row['RA'], row['DEC']))
    #     counter = counter + 1
    #     if counter == 10:
    #         break

    # Worst Galaxy logp
    # print("WORST GENERIC GALAXY LOGP")
    # _d = dfs.loc[dfs['galaxy']& dfs['generic']]
    # _d = _d.sort_values('logp_marg_corr')
    # counter = 0
    # for index, row in _d.iterrows():
    #     # print(row[selection])
    #     print(row['logp_marg_corr'],row['new_label'])
    #     print("http://skyserver.sdss.org/dr17/en/tools/quicklook/summary.aspx?plate={}&mjd={}&fiber={}".format(
    #         int(row['plate']), int(row['MJD']), int(row['fiber'])))
    #     print("https://www.legacysurvey.org/viewer-desi?ra={}&dec={}&zoom=16".format(
    #         row['RA'], row['DEC']))
    #     counter = counter + 1
    #     if counter == 10:
    #         break

    # Worst QSO recon error
    # print("WORST QSO Recon")
    # _d = dfs.loc[~dfs['galaxy']]
    # _d = _d.sort_values('recon_error',ascending=False)
    # counter = 0
    # for index, row in _d.iterrows():
    #     # print(row[selection])
    #     print(row['recon_error'],row['new_label'])
    #     print("http://skyserver.sdss.org/dr17/en/tools/quicklook/summary.aspx?plate={}&mjd={}&fiber={}".format(
    #         int(row['plate']), int(row['MJD']), int(row['fiber'])))
    #     print("https://www.legacysurvey.org/viewer-desi?ra={}&dec={}&zoom=16".format(
    #         row['RA'], row['DEC']))
    #     counter = counter + 1
    #     if counter == 5:
    #         break

    # Worst galaxy generic recon error
    # _d = dfs.loc[dfs['galaxy'] & dfs['generic']]
    # _d = _d.sort_values('recon_error',ascending=False)
    # counter = 0
    # for index, row in _d.iterrows():
    #     # print(row[selection])
    #     print(row['recon_error'],row['new_label'])
    #     print("http://skyserver.sdss.org/dr17/en/tools/quicklook/summary.aspx?plate={}&mjd={}&fiber={}".format(
    #         int(row['plate']), int(row['MJD']), int(row['fiber'])))
    #     print("https://www.legacysurvey.org/viewer-desi?ra={}&dec={}&zoom=16".format(
    #         row['RA'], row['DEC']))
    #     counter = counter + 1
    #     if counter == 10:
    #         break

    # GALAXY QSO
    # for s in selections:
    #     plt.clf()
    #     plt.hist([dfs[s][dfs.galaxy == True], dfs[s][dfs.galaxy == False]], bins=bins[s],
    #              label=['Galaxy', 'QSO'], density=True)
    #     plt.legend()
    #     plt.yscale('log')
    #     plt.xlabel(s)
    #     plt.show()

    # BROADLINE
    # for s in selections:
    #     plt.clf()
    #     plt.hist([dfs[s][dfs.broadline == True], dfs[s][dfs.broadline == False]], bins=bins[s],
    #              label=['Broadline', 'Normal'], density=True)
    #     plt.legend()
    #     plt.yscale('log')
    #     plt.xlabel(s)
    #     plt.show()

    # Galaxy subtypes
    # for s in selections:
    #     plt.clf()
    #     plt.hist([dfs.loc[dfs['generic'] & dfs['galaxy']][s], dfs.loc[dfs['AGN'] & dfs['galaxy']][s], dfs.loc[dfs['starburst'] & dfs['galaxy']][s], dfs.loc[dfs['starforming'] & dfs['galaxy']][s]], bins = bins[s],
    #                                                                          label = ['generic', 'AGN',
    #                                                                                   'starburst',
    #                                                                                   'starforming'], density = True)
    #     # print(dfs.new_label[numpy.logical_and(dfs.starforming == True, gal)].unique())
    #     # print(numpy.logical_and(dfs.starforming == True, gal).sum())
    #     plt.legend()
    #     plt.yscale('log')
    #     plt.xlabel(s)
    #     plt.title('galaxy')
    #     plt.show()

    plt.plot(dfs['logp_marg_corr'][dfs['galaxy']], dfs['recon_error'][dfs['galaxy']], '.', markersize=4, label='galaxy')
    plt.plot(dfs['logp_marg_corr'][~dfs['galaxy']], dfs['recon_error'][~dfs['galaxy']], '.', markersize=4, label='QSO')
    plt.xlim(-50, 25)
    plt.ylim(0, 40)
    plt.xlabel('logp_marg_corr')
    plt.ylabel('recon_error')
    plt.legend()
    plt.show()
    wfe

    # for s in selections:
    #     fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    # for i in range(16):
    #     _d = dfs.loc[dfs['new_label'] == i]
    # ax[i // 8].plot(_d['z'], _d[s], '.', label=i, alpha=1, markersize=4)
    #
    # ax[0].set_ylim((-60, -10))
    # ax[1].set_ylim((-60, -10))
    # ax[0].legend()
    # ax[1].legend()
    # plt.legend()
    # plt.show()
    # plt.legend()
    # wef
    #
    # dfs.loc[(dfs.new_label >= 8), 'new_label'] = 8
    # labels = numpy.sort(dfs['new_label'].unique())
    # for i in labels:
    #     _d = dfs.loc[dfs['new_label'] == i]
    # plt.plot(_d['z'], _d[s], '.', label=i, alpha=1, markersize=4)
    # plt.legend(loc=2)
    # plt.ylim((-45, -20))
    # plt.show()
    # wef
    # if s == 'logp_marg_corr':
    #     mi = numpy.min([h.min() for h in holder])
    #     b = numpy.linspace(mi, mi + 15, 5)
    #     b = numpy.linspace(-40, 25, 25)
    #     plt.hist(holder, bins=b, label=tags)
    #     plt.xlabel(s)
    #
    # elif s == 'recon_error':
    #     b = numpy.logspace(1e-5, 0.01, 5)
    #     b = numpy.linspace(0.3, 1, 20)
    #     # for i in range(len(holder)):
    #     #     holder[i]=holder[i]-1
    #     plt.hist(holder, bins=b, label=tags)
    #     # plt.xscale('log')
    #     plt.xlabel(s)
    #
    # plt.legend()
    # plt.yscale('log')
    # plt.xlabel(s)
    # plt.show()


def quickView(selection='logp_marg_corr'):
    print(selection)
    f = 'training_set_dec.csv'
    df = pandas.read_csv(f)
    # print(df.columns)
    ascending = dict()
    ascending['logp_marg_corr'] = True
    ascending['recon_error'] = False
    for i in range(2):
        print(i)
        _d = df.loc[df['new_label'] == i]
        _d = _d.sort_values(selection, ascending=ascending[selection])

        counter = 0
        for index, row in _d.iterrows():
            # print(row[selection])
            print("http://skyserver.sdss.org/dr17/en/tools/quicklook/summary.aspx?plate={}&mjd={}&fiber={}".format(
                int(row['plate']), int(row['MJD']), int(row['fiber'])))
            print("https://www.legacysurvey.org/viewer-desi?ra={}&dec={}&zoom=16".format(
                row['RA'], row['DEC']))

            counter = counter + 1
            if counter == 5:
                break
            # print("https://dr17.sdss.org/optical/spectrum/view?plate={}&mjd={}&fiber={}".format(
            #     int(row['plate']), int(row['MJD']), int(row['fiber'])))


def compareList():
    f = 'test_set_no_QSO_relabeled_retrained.csv'
    df = pandas.read_csv(f)
    for i in range(3, 4):
        print(i)
        _d = df.loc[df['subclass'] == i]
        _d = _d.sort_values('logp')
        counter = 0
        for index, row in _d.iterrows():
            print(int(row['plate']), int(row['MJD']), int(row['fiber']), row['logp'])
            counter = counter + 1
            if counter == 12:
                break
            # print("https://dr17.sdss.org/optical/spectrum/view?plate={}&mjd={}&fiber={}".format(
            #     int(row['plate']), int(row['MJD']), int(row['fiber'])))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    outliers()
    # tails()
    # supernova()
    # addDec()
    # quickView('logp_marg_corr')
    # quickView('recon_error')
    # compareList()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
