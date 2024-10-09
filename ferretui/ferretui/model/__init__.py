try:
    from .language_model.ferret_llama import FerretLlamaForCausalLM, FerretConfig
    from .language_model.ferret_mpt import FerretMptForCausalLM, FerretMptConfig
    from .language_model.ferret_gemma import FerretGemmaForCausalLM, FerretGemmaConfig
except:
    print('cant import')
    pass