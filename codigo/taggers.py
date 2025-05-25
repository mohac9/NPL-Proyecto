import nltk
import parser as ps

if __name__ == "__main__":
    #Tagging the info 
    pokemons = ps.read_csv()
    
    tagged_info = []
    for pokemon in pokemons:
        info = pokemon.get_info()
        print(info)
        tagged_info = nltk.pos_tag(info)
        tagged_info.append(tagged_info)
        

    