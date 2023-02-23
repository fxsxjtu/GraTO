from new_train_newmeasure import NAS_trainer

trainer = NAS_trainer(nhid=256, lr=0.045263157894736845, weight_decay=0.0005, device='cuda:0', layers=2,
                     path='./', data_name='citeseer',derive='DARTS_pubmed',alpha=1)  # set your own parameter and get your result
trainer.train_session()



