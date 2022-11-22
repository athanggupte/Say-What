def soft_update(source_model, target_model, tau):
    for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)