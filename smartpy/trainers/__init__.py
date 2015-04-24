import os

from smartpy.trainers.status import Status


# def load(experiment_path):
#     # Load command to resume
#     data_dir = args.experiment
#     launch_command = pickle.load(open(pjoin(args.experiment, "command.pkl")))
#     command_to_resume = sys.argv[1:sys.argv.index('resume')] + launch_command
#     args = parser.parse_args(command_to_resume)

#     args.subcommand = "resume"

#     print "Loading dataset..."
#     dataset = Dataset(args.dataset)

#     print "Building model..."
#     nade = models.factory("NADE", input_size=dataset.input_size, hyperparams=vars(args))

#     from smartpy.misc import weights_initializer
#     weights_initialization_method = weights_initializer.factory(**vars(args))
#     nade.initialize(weights_initialization_method)

#     ### Build trainer ###
#     optimizer = optimizers.factory(args.optimizer, loss=nade.mean_nll_loss, **vars(args))
#     optimizer.add_update_rule(*args.update_rules)

#     trainer = Trainer(model=nade, datasets=[dataset.trainset], optimizer=optimizer)

#     # Add stopping criteria
#     if args.max_epochs is not None:
#         # Stop when max number of epochs is reached.
#         print "Will train {0} for a total of {1} epochs.".format(args.model, args.max_epochs)
#         trainer.add_stopping_criterion(tasks.MaxEpochStopping(args.max_epochs))

#     # Print time for one epoch
#     trainer.add_task(tasks.PrintEpochDuration())
#     trainer.add_task(tasks.AverageObjective(trainer))
#     avg_nll_on_valid = tasks.AverageNLL(nade.get_nll, dataset.validset, batch_size=100)
#     trainer.add_task(tasks.Print(avg_nll_on_valid, msg="Average NLL on the validset: {0}"))


#     # Do early stopping bywatching the average NLL on the validset.
#     if args.lookahead is not None:
#         print "Will train {0} using early stopping with a lookahead of {1} epochs.".format(args.model, args.lookahead)
#         save_task = tasks.SaveTraining(trainer, savedir=data_dir)
#         early_stopping = tasks.EarlyStopping(avg_nll_on_valid, args.lookahead, save_task, eps=1e-6)
#         trainer.add_stopping_criterion(early_stopping)
#         trainer.add_task(early_stopping)

#     # Add a task to save the whole training process
#     if args.save_frequency < np.inf:
#         save_task = tasks.SaveTraining(trainer, savedir=data_dir, each_epoch=args.save_frequency)
#         trainer.add_task(save_task)

#     if args.subcommand == "resume":
#         print "Loading existing trainer..."
#         trainer.load(data_dir)

#     trainer.run()