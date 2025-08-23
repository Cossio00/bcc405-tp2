import torch
import torch.nn as nn
import torch.optim as optim
import copy

class DLGAttack:
    def __init__(self, model, target_gradients, target_labels=None, lr=0.1, optimizer_name="L-BFGS", num_iters=10000, num_restarts=5, device="cpu"):
        self.model = copy.deepcopy(model).to(device).train()
        for param in self.model.parameters():
            param.requires_grad_(True)
        self.target_gradients = [g.detach().to(device) for g in target_gradients]
        self.target_labels = target_labels
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.num_iters = num_iters
        self.num_restarts = num_restarts
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.params = tuple(p for p in self.model.parameters())
        
        # Determinar o tamanho do batch a partir dos gradientes alvo
        self.batch_size = target_gradients[0].shape[0] if target_gradients[0].dim() > 1 else 1

    def _init_data(self, shape, num_classes=10):
        batch_size = shape[0] if len(shape) > 3 else 1
        
        # Inicializar dados dummy com o tamanho correto do batch
        x_hat = torch.nn.Parameter(torch.randn(*shape, device=self.device) * 0.1)
        
        if self.target_labels is not None:
            # Usar os rótulos alvo se fornecidos
            y_hat = torch.tensor(self.target_labels, device=self.device, dtype=torch.long)
        else:
            # Inicializar logits para todos os exemplos do batch
            y_hat = torch.nn.Parameter(torch.randn(batch_size, num_classes, device=self.device) * 0.01)
        
        return x_hat, y_hat

    def _get_optimizer(self, params):
        if self.optimizer_name.lower() == "adam":
            return optim.Adam(params, lr=self.lr)
        else:
            return optim.LBFGS(params, lr=self.lr, max_iter=20)

    @torch.enable_grad()
    def _compute_gradients(self, x_hat, y_hat):
        out = self.model(x_hat)
        
        if y_hat.dim() == 1:  # Rótulos fixos (já são inteiros)
            loss = self.criterion(out, y_hat)
        else:  # Logits otimizáveis
            # Calcular loss para cada exemplo no batch
            loss = 0
            for i in range(out.shape[0]):
                # Usar softmax dos logits como pesos para a loss
                y_probs = torch.softmax(y_hat[i], dim=-1)
                loss += (out[i] * y_probs).sum()
        
        g_hat = torch.autograd.grad(loss, self.params, create_graph=True, retain_graph=True)
        return g_hat, loss

    def _grad_loss(self, g_hat):
        total_loss = 0
        for gh, gt in zip(g_hat, self.target_gradients):
            if gh is not None and gt is not None:
                total_loss += ((gh - gt) ** 2).sum()
        return total_loss

    def run(self, input_shape=(1, 1, 28, 28), num_classes=10):
        best_result = None
        best_loss_val = float("inf")
        best_y_recon = None
        
        for restart in range(self.num_restarts):
            x_hat, y_hat = self._init_data(input_shape, num_classes)
            params = [x_hat] if isinstance(y_hat, torch.Tensor) and y_hat.dim() == 1 else [x_hat, y_hat]
            optimizer = self._get_optimizer(params)

            def closure():
                optimizer.zero_grad(set_to_none=True)
                g_hat, pred_loss = self._compute_gradients(x_hat, y_hat)
                
                # Adicionar penalidades para incentivar rótulos discretos
                if y_hat.dim() > 1:  # Logits otimizáveis
                    y_softmax = torch.softmax(y_hat, dim=-1)
                    discrete_penalty = (1.0 - y_softmax.max(dim=-1)[0]).sum()
                    entropy_penalty = -(y_softmax * torch.log(y_softmax + 1e-8)).sum(dim=-1).mean()
                    loss_tensor = self._grad_loss(g_hat) + 10.0 * discrete_penalty + 5.0 * entropy_penalty + 1e-5 * x_hat.pow(2).sum()
                else:
                    loss_tensor = self._grad_loss(g_hat) + 1e-5 * x_hat.pow(2).sum()
                
                loss_tensor.backward()
                return loss_tensor

            if isinstance(optimizer, optim.LBFGS):
                optimizer.step(closure)
            else:
                for it in range(self.num_iters):
                    loss_tensor = closure()
                    optimizer.step()
                    if it % 500 == 0:
                        print(f"[Restart {restart}] Iter {it}/{self.num_iters} - Loss: {float(loss_tensor.detach()):.4f}")
            
            with torch.no_grad():
                g_hat_final, _ = self._compute_gradients(x_hat, y_hat)
                final_loss_val = float(self._grad_loss(g_hat_final).detach())
                
                # Determinar os rótulos reconstruídos
                if y_hat.dim() > 1:  # Logits otimizáveis
                    y_recon = y_hat.argmax(dim=-1)  # Obter argmax para cada exemplo
                    y_recon = y_recon.cpu()  # Mover para CPU para evitar problemas
                else:  # Rótulos fixos
                    y_recon = y_hat.cpu()  # Já são inteiros
                
                # Verificar se convergiu para rótulos razoáveis
                if y_hat.dim() > 1:
                    y_softmax = torch.softmax(y_hat, dim=-1)
                    max_probs = y_softmax.max(dim=-1)[0]
                    if any(max_probs < 0.8):  # Se algum não convergiu bem
                        print(f"Warning: Some softmax probabilities are low: {max_probs.tolist()}")
            
            if final_loss_val < best_loss_val:
                best_loss_val = final_loss_val
                best_result = x_hat.detach().clone()
                best_y_recon = y_recon.clone()
        
        # Ajustar formato da imagem reconstruída
        if best_result.dim() == 3:  # Se for (C, H, W), adicionar dimensão de batch
            best_result = best_result.unsqueeze(0)
        
        return (best_result, best_y_recon), best_loss_val