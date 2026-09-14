"""Declared marginal structure for the Spanish 2021 census.

'blocks' is a tree of blocks chosen by measured pairwise association: relatives hang off their
SEXO_* column, the nucleus and the spouse off TIPOPER, where the declared rules cut the bags down.
"""

STRUCTURES = {
    'blocks': [
        # Person and spouse
        ['SEXO', 'TIPOPER', 'ECIVIL', 'SEXO_CON', 'RELA'],
        ['TIPOPER', 'SEXO_CON', 'ECIVL_CON', 'RELA_CON', 'SITU_CON', 'ESREAL_CON_GR5'],
        ['SEXO_CON', 'PNACIM_CON_GR9', 'NACIO_CON_GR10'],
        ['SEXO_CON', 'VAREDAD_CON'],
        # Nucleus and household
        ['TIPOPER', 'TIPO_NUC', 'TIPO_PAR_NUC1', 'TIPO_PAR_NUC2'],
        ['TIPOPER', 'TIPO_NUC', 'TAM_NUC', 'NHIJOS_NUC'],
        ['TIPO_NUC', 'TAM_NUC', 'ESTRUC_HOG', 'TAM_HOG'],
        ['ESTRUC_HOG', 'TAM_HOG', 'TIPO_HOG', 'NUC_HOG'],
        # Parents
        ['TIPOPER', 'SEXO_MAD', 'SEXO_PAD'],
        ['SEXO_MAD', 'ECIVL_MAD', 'RELA_MAD', 'SITU_MAD'],
        ['SEXO_MAD', 'ESREAL_MAD_GR5', 'PNACIM_MAD_GR9', 'NACIO_MAD_GR10'],
        ['SEXO_MAD', 'VAREDAD_MAD'],
        ['SEXO_PAD', 'ECIVL_PAD', 'RELA_PAD', 'SITU_PAD'],
        ['SEXO_PAD', 'ESREAL_PAD_GR5', 'PNACIM_PAD_GR9', 'NACIO_PAD_GR10'],
        ['SEXO_PAD', 'VAREDAD_PAD'],
        # Dwelling
        ['TAM_HOG', 'SUP_OCU_VIV', 'SUP_VIV'],
        ['SUP_VIV', 'TIPO_EDIF_VIV', 'ANO_CONS', 'TENEN_VIV'],
        ['TIPO_EDIF_VIV', 'NPLANTAS_SOBRE_EDIF', 'NPLANTAS_BAJO_EDIF', 'TIPO_MUN_DEGURBA'],
        # Activity, education and work
        ['RELA', 'VAREDAD', 'ESREAL_CNEDA'],
        ['RELA', 'SITU', 'LTRAB'],
        ['SITU', 'OCU63'],
        ['OCU63', 'ACT89'],
        ['LTRAB', 'CPRO_TRAB'],
        ['RELA', 'ESCUR', 'LEST', 'TESCUR'],
        ['LEST', 'CPRO_EST'],
        ['ESCUR', 'ESCUR2'],
        # Birth, nationality and migration
        ['SEXO', 'MNAC'],
        ['SEXO', 'RESI_NACIM'],
        ['RESI_NACIM', 'PNACIM', 'VARANOE'],
        ['PNACIM', 'PAIS_PR'],
        ['RESI_NACIM', 'PNACIO'],
        ['RESI_NACIM', 'RESI_ANT', 'CPRO_NAC'],
        ['RESI_ANT', 'CPRO_ANT', 'VARANOM'],
        ['RESI_ANT', 'PAIS_ANT'],
        ['CPRO_ANT', 'PROV_PR'],
        ['PROV_PR', 'ANOP', 'VARANOC'],
        ['VARANOM', 'VARANORES', 'RESI_UNANO', 'RESI_DANO'],
        ['RESI_DANO', 'PAIS_DANO'],
        ['RESI_UNANO', 'PAIS_UNANO'],
        ['CPRO_NAC', 'CPRO_UNANO'],
        ['CPRO_UNANO', 'CPRO_DANO'],
    ],
}

DEFAULT = 'blocks'
