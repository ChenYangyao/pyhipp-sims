from pyhipp_sims import sims

def test_tng():
    sim_info = sims.predefined['tng']
    assert sim_info.name == 'tng'
    
    sim_info = sims.predefined['tng']
    print(sim_info.box_size, sim_info.mass_table)
    
    cosm = sim_info.cosmology
    print(cosm.hubble, cosm.omega_l0)
    
    assert cosm.distances.comoving_at(z=[0.0, 1.0, 2.0]).size == 3
    
    astropy_cosm = cosm.astropy_model
    print(astropy_cosm)