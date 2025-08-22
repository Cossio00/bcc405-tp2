import torch
import torch.nn as nn
import torch.optim as optim
import copy

class DLGAttack:
    def __init__(self, model, target_gradients, target_labels=None,
                 lr=0.1, optimizer_name="L-BFGS", num_iters=2000,
                 num_restarts=5, device="cpu"):
        """
        Args:
            model: modelo alvo (torch.nn.Module)
            target_gradients: lista de tensores (gradientes vazados g*)
            target_labels: rótulo real (opcional, usado em iDLG)
            lr: taxa de aprendizado
            optimizer_name: "L-BFGS" ou "Adam"
            num_iters: iterações de otimização
            num_restarts: quantos reinícios aleatórios (pega melhor)
        """
        self.model = copy.deepcopy(model).to(device).eval()  # eval() evita batchnorm/dropout variando
        self.target_gradients = [g.detach().to(device) for g in target_gradients]
        self.target_labels = target_labels
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.num_iters = num_iters
        self.num_restarts = num_restarts
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        # Cache dos parâmetros como tupla para manter ordem estável
        self.params = tuple(p for p in self.model.parameters())

    def _init_data(self, shape, num_classes=10):
        """Inicializa x_hat (imagem) e y_hat (logits/rótulo)."""
        x_hat = torch.randn(shape, device=self.device, requires_grad=True)

        if self.target_labels is not None:
            # rótulo conhecido (DLG com label conhecido ou iDLG)
            y_hat = torch.tensor([self.target_labels], device=self.device)
        else:
            # logits livres (otimizáveis)
            y_hat = torch.randn((1, num_classes), device=self.device, requires_grad=True)

        return x_hat, y_hat

    def _get_optimizer(self, params):
        if self.optimizer_name.lower() == "adam":
            return optim.Adam(params, lr=self.lr)
        else:
            return optim.LBFGS(params, lr=self.lr, max_iter=20, line_search_fn=None)

    @torch.enable_grad()
    def _compute_gradients(self, x_hat, y_hat):
        """
        Computa gradientes simulados g_hat a partir de x_hat e y_hat,
        mantendo grafo para permitir backprop através de g_hat.
        """
        # forward
        out = self.model(x_hat)
        if y_hat.ndim == 1:  # classe conhecida
            loss = self.criterion(out, y_hat)
        else:                # logits como variável (soft labels)
            loss = torch.mean(torch.sum(-y_hat * torch.log_softmax(out, dim=-1), dim=-1))

        # gradientes dos parâmetros com create_graph=True (segunda ordem)
        g_hat = torch.autograd.grad(
            loss,
            self.params,
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )
        return g_hat, loss

    def _grad_loss(self, g_hat):
        """Distância L2 entre g* e g_hat (soma sobre todos os parâmetros)."""
        return sum(((gh - gt) ** 2).sum() for gh, gt in zip(g_hat, self.target_gradients))

    def run(self, input_shape=(1, 1, 28, 28), num_classes=10):
        best_result = None
        best_loss_val = float("inf")

        for restart in range(self.num_restarts):
            x_hat, y_hat = self._init_data(input_shape, num_classes)
            params = [x_hat] if y_hat.ndim == 1 else [x_hat, y_hat]
            optimizer = self._get_optimizer(params)

            def closure():
                optimizer.zero_grad(set_to_none=True)
                g_hat, _ = self._compute_gradients(x_hat, y_hat)
                loss_tensor = self._grad_loss(g_hat)   # este tensor tem grad_fn
                loss_tensor.backward()
                return loss_tensor

            if isinstance(optimizer, optim.LBFGS):
                # LBFGS chamará o closure múltiplas vezes internamente
                loss_tensor = optimizer.step(closure)
                loss_val = float(loss_tensor.detach())
            else:
                # Loop manual para Adam
                for it in range(self.num_iters):
                    loss_tensor = closure()
                    optimizer.step()
                    if it % 500 == 0:
                        print(f"[Restart {restart}] Iter {it}/{self.num_iters} - Loss: {float(loss_tensor.detach()):.4f}")
                loss_val = float(loss_tensor.detach())

            # Calcula loss final deste restart (com grafo novo, mas só para avaliar)
            with torch.no_grad():
                g_hat_final, _ = self._compute_gradients(x_hat, y_hat)
                final_loss_val = float(self._grad_loss(g_hat_final).detach())

            if final_loss_val < best_loss_val:
                best_loss_val = final_loss_val
                best_result = (x_hat.detach().clone(), y_hat.detach().clone())

        return best_result, best_loss_val
