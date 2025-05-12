import csv

class Pokemon:
    def __init__(self,id,name,types,info):
        self.id = id
        self.name = name
        self.types = types
        self.info = info
    
    #Getters
    def get_id(self):
        return self.id
    def get_name(self):
        return self.name
    def get_types(self):
        return self.types
    def get_info(self):
        return self.info
    
def read_csv(file_path='pokedex.csv'):
    pokemons = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            id = row[0]
            name = row[1]
            types = row[10]
            info = row[12]
            pokemon = Pokemon(id, name, types, info)
            pokemons.append(pokemon)
            
    return pokemons

if __name__ == "__main__":
    pokemons = read_csv()
    for pokemon in pokemons:
        print(f"ID: {pokemon.get_id()}, Name: {pokemon.get_name()}, Types: {pokemon.get_types()}, Info: {pokemon.get_info()}")
