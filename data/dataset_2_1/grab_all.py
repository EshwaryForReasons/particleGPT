import dictionary

def tokenize_particle(particle):
    if particle[0] == 'EVENT_START':
        return
    elif particle[0] == 'EVENT_END':
        return
    elif particle[0] == 'PADDING':
        return
    
    pdgid = int(particle[0])
    dictionary.particle_id_to_index(pdgid)

def tokenize_data():
    input_file = open('data.csv', 'r')

    for event in input_file:
        particles = event.split(';')
        
        for particle in particles:
            data = particle.split()
            tokenize_particle(data)
    
    input_file.close()
    
tokenize_data()